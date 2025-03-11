//! Posit implementation

const std = @import("std");
const Int = std.meta.Int;
const IntFittingRange = std.math.IntFittingRange;

//TODO: change assert in comptime assert

fn PackType(comptime TupleType: type) type {
    if (@bitSizeOf(TupleType) == 0) return u0;
    const struct_def = @typeInfo(TupleType).@"struct";
    var fields: [struct_def.fields.len]std.builtin.Type.StructField = undefined;
    for (struct_def.fields, 0..) |field, i| {
        fields[i] = field;
        fields[i].alignment = 0;
        fields[i].is_comptime = false;
    }
    return @Type(.{
        .@"struct" = .{
            .layout = .@"packed",
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

fn pack(tuple: anytype) PackType(@TypeOf(tuple)) {
    if (@bitSizeOf(@TypeOf(tuple)) == 0) {
        return @as(u0, 0);
    } else {
        var out: PackType(@TypeOf(tuple)) = undefined;
        const fields = @typeInfo(@TypeOf(tuple)).@"struct".fields;
        inline for (fields) |field| {
            @field(out, field.name) = @field(tuple, field.name);
        }
        return out;
    }
}

const Sign = enum { positive, negative };

fn GetFloatStructure(comptime T: type) type {
    std.debug.assert(@typeInfo(T) == .float);
    const frac_bits = std.math.floatMantissaBits(T);
    return struct {
        const Fraction = Int(.unsigned, frac_bits);
        const Exponent = IntFittingRange(std.math.floatExponentMin(T), std.math.floatExponentMax(T));
        fraction: Fraction,
        exponent: Exponent,
        sign: Sign,
    };
}

fn decodeFloat(comptime T: type, x: T) union(enum) {
    zero,
    inf,
    regular: GetFloatStructure(T),
} {
    std.debug.assert(@typeInfo(T) == .float);
    if (std.math.isNan(x)) return .inf;
    if (x == 0) return .zero; //works for negative zero
    const FloatStructure = GetFloatStructure(T);
    const exp_size = std.math.floatExponentBits(T);
    const FloatPackedStructure = packed struct { frac: FloatStructure.Fraction, exp: Int(.unsigned, exp_size), sign: bool };
    const xPacked: FloatPackedStructure = @bitCast(x);
    const frac = xPacked.frac;
    const exponent = @as(FloatStructure.Exponent, @bitCast(xPacked.exp)) -% std.math.floatExponentMax(T);
    return .{ .regular = .{
        .fraction = frac,
        .exponent = exponent,
        .sign = if (xPacked.sign) .negative else .positive,
    } };
}

fn encodeFloat(comptime Float: type, comptime T: type, comptime M: type, frac: T, exp: M, sign: Sign) Float {
    std.debug.assert(@typeInfo(Float) == .float);
    std.debug.assert(@typeInfo(T) == .int);
    std.debug.assert(@typeInfo(T).int.signedness == .unsigned);
    std.debug.assert(@typeInfo(M) == .int);
    const FloatStructure = GetFloatStructure(Float);
    if (exp >= std.math.floatExponentMax(Float)) return std.math.floatMax(Float);
    if (exp < std.math.floatExponentMin(Float)) return std.math.floatMin(Float);
    const exp_size = std.math.floatExponentBits(Float);
    const frac_size = std.math.floatMantissaBits(Float);
    var exp_bits: Int(.unsigned, exp_size) = @bitCast(@as(FloatStructure.Exponent, @intCast(exp)) +% std.math.floatExponentMax(Float));
    const sign_bit: u1 = if (sign == .positive) 0 else 1;
    const shf_amt = @bitSizeOf(T) - frac_size;
    var frac_bits: FloatStructure.Fraction = undefined;
    if (@bitSizeOf(T) == 0) {
        frac_bits = 0;
    } else if (shf_amt <= 0) {
        frac_bits = frac;
        frac_bits <<= -shf_amt;
    } else {
        frac_bits = @truncate(frac >> shf_amt);
        const out_bits = frac << frac_size;
        const half = 1 << @bitSizeOf(T) - 1;
        if (out_bits == half and frac_bits & 1 != 0) frac_bits += 1;
        if (out_bits > half) {
            frac_bits +%= 1;
            if (frac_bits == 0) {
                exp_bits += 1;
            }
        }
    }
    return @bitCast(pack(struct { FloatStructure.Fraction, Int(.unsigned, exp_size), u1 }{ frac_bits, exp_bits, sign_bit }));
}

/// Posit implementation
pub fn Posit(comptime nbit: comptime_int, comptime es: comptime_int) type {
    return packed struct {
        pub const max_regime = nbit - 2;
        pub const max_exponent = max_regime << es;
        pub const Self = @This();
        pub const IntRepr = Int(.signed, nbit);
        pub const UIntRepr = Int(.unsigned, nbit);
        pub const Regime = IntFittingRange(-max_regime, max_regime);
        pub const Fraction = Int(.unsigned, @max(0, nbit - es - 3));
        pub const Es = Int(.unsigned, es);
        pub const Exponent = Int(.signed, @bitSizeOf(Regime) + es);

        // important values
        pub const inf = Self{ .bits = 1 << (nbit - 1) };
        pub const zero = Self{ .bits = 0 };
        pub const one = Self{ .bits = 1 << (nbit - 2) };
        pub const max_pos = inf.prior();
        pub const min_pos = zero.next();

        //mathematical values
        pub const pi = fromFloat(f128, std.math.pi);
        pub const tau = fromFloat(f128, std.math.tau);

        bits: UIntRepr,

        pub const DecodeType = union(enum) {
            regular: struct {
                sign: Sign,
                exp: Exponent,
                frac: Fraction,
            },
            zero,
            inf,
        };

        pub fn negate(self: Self) Self {
            const signed_bits: IntRepr = @bitCast(self.bits);
            return .{ .bits = @bitCast(-%signed_bits) };
        }

        pub fn abs(self: Self) Self {
            return if (self.isNegative()) self.negate() else self;
        }

        pub fn sign(self: Self) Self {
            if (self.bits == zero.bits) return zero;
            if (self.bits == inf.bits) return inf;
            return if (self.isNegative()) one.negate() else one;
        }

        pub fn isNegative(self: Self) bool {
            return @as(IntRepr, @bitCast(self.bits)) < 0;
        }

        pub fn next(self: Self) Self {
            return .{ .bits = self.bits +% 1 };
        }

        pub fn prior(self: Self) Self {
            return .{ .bits = self.bits -% 1 };
        }

        pub fn floor(self: Self) Self {
            if (self.isNegative()) {
                return self.ceilFromZero();
            } else {
                return self.floorToZero();
            }
        }

        pub fn ceil(self: Self) Self {
            if (self.isNegative()) {
                return self.floorToZero();
            } else {
                return self.ceilFromZero();
            }
        }

        pub fn ceilFromZero(self: Self) Self {
            return switch (self.decode()) {
                .inf => inf,
                .zero => zero,
                .regular => |self_regular| blk: {
                    const exp = self_regular.exp;
                    if (exp < 0) break :blk one;
                    var mask = ~@as(Fraction, 0);
                    mask = std.math.shr(Fraction, mask, exp);
                    const frac = self_regular.frac & ~mask;
                    var out = encode(Fraction, Exponent, frac, exp, .positive);
                    if (self_regular.frac & mask != 0) out = out.add(one);
                    break :blk switch (self_regular.sign) {
                        .positive => out,
                        .negative => out.negate(),
                    };
                },
            };
        }

        pub fn floorToZero(self: Self) Self {
            return switch (self.decode()) {
                .inf => inf,
                .zero => zero,
                .regular => |self_regular| blk: {
                    const exp = self_regular.exp;
                    if (exp < 0) break :blk zero;
                    var mask = ~@as(Fraction, 0);
                    mask = std.math.shr(Fraction, mask, exp);
                    const frac = self_regular.frac & ~mask;
                    var out = encode(Fraction, Exponent, frac, exp, .positive);
                    break :blk switch (self_regular.sign) {
                        .positive => out,
                        .negative => out.negate(),
                    };
                },
            };
        }

        pub fn eq(self: Self, other: Self) bool {
            return self.bits == other.bits;
        }

        pub fn nEq(self: Self, other: Self) bool {
            return self.bits != other.bits;
        }

        pub fn gr(self: Self, other: Self) bool {
            return @as(IntRepr, @bitCast(self.bits)) > @as(IntRepr, @bitCast(other.bits));
        }

        pub fn grEq(self: Self, other: Self) bool {
            return @as(IntRepr, @bitCast(self.bits)) >= @as(IntRepr, @bitCast(other.bits));
        }

        pub fn le(self: Self, other: Self) bool {
            return @as(IntRepr, @bitCast(self.bits)) < @as(IntRepr, @bitCast(other.bits));
        }

        pub fn leEq(self: Self, other: Self) bool {
            return @as(IntRepr, @bitCast(self.bits)) <= @as(IntRepr, @bitCast(other.bits));
        }

        pub fn decodeOld(self: Self) DecodeType {
            if (self.eq(zero)) return .zero;
            if (self.eq(inf)) return .inf;
            const bits = self.abs().bits;
            const bits1: Int(.unsigned, nbit - 1) = @intCast(bits);
            const bits2: packed struct { bits: Int(.unsigned, nbit - 2), regType: u1 } = @bitCast(bits1);
            const regime_len_minus_one = @clz(if (bits2.regType == 0) bits2.bits else ~bits2.bits);
            const bits3 = std.math.shl(Int(.unsigned, nbit - 2), bits2.bits, regime_len_minus_one);
            const regime: Regime = if (bits2.regType == 1) @as(Regime, regime_len_minus_one) else ~@as(Regime, regime_len_minus_one);
            const bits4: Int(.unsigned, @max(0, nbit - 3)) = @truncate(bits3);
            const Pad = Int(.unsigned, @max(0, es - @max(0, nbit - 3)));
            const comp: packed struct { frac: Fraction, exp: Exponent } = @bitCast(
                pack(struct { Pad, @TypeOf(bits4), Regime }{ 0, bits4, regime }),
            );
            return .{ .regular = .{
                .sign = if (self.isNegative()) .negative else .positive,
                .exp = comp.exp,
                .frac = comp.frac,
            } };
        }

        pub fn decodeStraigth(self: Self) DecodeType {
            if (self.eq(zero)) return .zero;
            if (self.eq(inf)) return .inf;
            const p_sign: Sign = if (self.isNegative()) .negative else .positive;
            var bits = self.abs().bits;
            bits <<= 1;
            const xor_mask = @as(IntRepr, @bitCast(bits)) >> (nbit - 1); //aritmetic shift;
            const clz_bits = bits ^ @as(UIntRepr, @bitCast(xor_mask));
            const regime_sign: u1 = @intFromBool(xor_mask < 0);
            const regime_len: IntFittingRange(0, nbit - 1) = @intCast(@clz(clz_bits));
            bits <<= regime_len;
            bits <<= 1;
            const frac: Fraction = @truncate(bits >> 3);
            const MachExp = IntCeil(Exponent);
            const regime_len_minus_one = @as(MachExp, (regime_len - 1));
            const regime = if (regime_sign == 1) regime_len_minus_one else ~regime_len_minus_one;
            const exp = @as(MachExp, regime) << es | @as(Int(.unsigned, @truncate(es)), if (es != 0) bits >> (nbit - es) else 0);
            return .{ .regular = .{
                .sign = p_sign,
                .exp = exp,
                .frac = frac,
            } };
        }

        pub fn decodeMask(self: Self) DecodeType {
            if (self.eq(zero)) return .zero;
            if (self.eq(inf)) return .inf;
            const p_sign: Sign = if (self.isNegative()) .negative else .positive;
            var bits = self.abs().bits;
            bits <<= 1;
            var xor_mask: IntRepr = @bitCast(bits); //aritmetic shift;
            xor_mask >>= (nbit - 1);
            const xor_umask: UIntRepr = @bitCast(xor_mask);
            const clz_bits = bits ^ xor_umask;
            //const regType: u1 = @boolToInt(xor_mask<0);
            const regime_len: IntFittingRange(0, nbit - 1) = @intCast(@clz(clz_bits));
            bits <<= regime_len;
            bits <<= 1;
            const frac: Fraction = @truncate(bits >> 3);
            const MachEspType = IntCeil(Exponent);
            const regime_len_minus_one: MachEspType = regime_len - 1;
            const regime = regime_len_minus_one ^ ~xor_mask;
            //const regime = if (regType == 1) regime_len_minus_one else ~regime_len_minus_one;
            const exp = regime << es | @as(Int(.unsigned, @truncate(es)), if (es != 0) bits >> (nbit - es) else 0);
            return .{ .regular = .{
                .sign = p_sign,
                .exp = exp,
                .frac = frac,
            } };
        }

        pub fn decodeLoop(self: Self) DecodeType {
            if (self.eq(zero)) return .zero;
            if (self.eq(inf)) return .inf;
            const p_sign: Sign = if (self.isNegative()) .negative else .positive;
            var bits = self.abs().bits;
            var exp: Exponent = undefined;
            if (bits & (1 << nbit - 2) != 0) {
                exp = -(1 << es);
                while (bits & (1 << nbit - 2) != 0) {
                    exp += (1 << es);
                    bits <<= 1;
                }
            } else {
                exp = 0;
                while (bits & (1 << nbit - 2) == 0) {
                    exp -= (1 << es);
                    bits <<= 1;
                }
            }
            const frac: Fraction = @truncate(bits >> 1);
            const es_bits: Es = @truncate(bits >> nbit - es - 2);
            exp = exp | @as(Exponent, es_bits);
            return .{ .regular = .{
                .sign = p_sign,
                .exp = exp,
                .frac = frac,
            } };
        }

        pub fn decode(self: Self) DecodeType {
            return self.decodeOld();
        }

        pub fn encode(comptime T: type, comptime M: type, frac: T, exp: M, out_sign: Sign) Self {
            std.debug.assert(@typeInfo(T) == .int);
            std.debug.assert(@typeInfo(M) == .int);
            std.debug.assert(@typeInfo(T).int.signedness == .unsigned);
            //std.debug.assert (@typeInfo(M).int.signedness == .signed);
            if (exp < -max_exponent) return if (out_sign == .positive) min_pos else min_pos.negate();
            if (exp >= max_exponent) return if (out_sign == .positive) max_pos else max_pos.negate();
            const out_exp: Exponent = @intCast(exp);
            const comp: packed struct { es: Es, reg: Regime } = @bitCast(out_exp);
            const regime = comp.reg;
            const es_bits = comp.es;
            const regime_len_minus_one: Int(.unsigned, @bitSizeOf(Regime) - 1) = @intCast(if (regime >= 0) regime else ~regime);
            var reg_mask: UIntRepr = 1;
            if (regime >= 0) {
                reg_mask = ~reg_mask;
            }
            const packed_structure = pack(struct { T, Es, UIntRepr }{ frac, es_bits, reg_mask });
            var packed_bits: Int(.unsigned, @bitSizeOf(T) + es + nbit) = @bitCast(packed_structure);

            packed_bits <<= nbit - 2 - regime_len_minus_one;

            const packed_result: packed struct {
                discarded: Int(.unsigned, @bitSizeOf(T) + es + 1),
                valid: Int(.unsigned, nbit - 1),
            } = @bitCast(
                packed_bits,
            );
            var bits: UIntRepr = packed_result.valid;
            //ora l'approssimazione
            const discarded_bits = packed_result.discarded;
            //geometric rounding in twilight zone https://gitlab.com/cerlane/SoftPosit/-/issues/14
            const rounding_mode: enum { arithmetic, geometric } = .geometric;
            var half: @TypeOf(discarded_bits) = 1 << @bitSizeOf(@TypeOf(discarded_bits)) - 1;

            comptime switch (rounding_mode) {
                .arithmetic => {
                    const half_packed = pack(struct { T, Es, UIntRepr }{ 1 << (@bitSizeOf(T) - 1), es_bits, 0 });
                    var half_wide: Int(.unsigned, @bitSizeOf(@TypeOf(half_packed))) = @bitCast(half_packed);
                    half_wide <<= nbit - 2 - regime_len_minus_one;
                    const half_candidate: Int(.unsigned, @bitSizeOf(@TypeOf(discarded_bits))) = @truncate(half_wide);
                    if (half_candidate != 0) half = half_candidate;
                },
                .geometric => {},
            };

            if (discarded_bits > half) bits += 1;
            if (discarded_bits == half and bits & 1 != 0) bits += 1;

            var out = Self{ .bits = bits };
            if (out_sign == .negative) {
                out = out.negate();
            }
            return out;
        }

        pub fn toPosit(self: Self, comptime T: type) T {
            return switch (self.decode()) {
                .zero => T.zero,
                .inf => T.inf,
                .regular => |v| T.encode(Fraction, Exponent, v.frac, v.exp, v.sign),
            };
        }

        pub fn toFloat(self: Self, comptime Float: type) Float {
            std.debug.assert(@typeInfo(Float) == .float);
            return switch (self.decode()) {
                .zero => 0,
                .inf => std.math.nan(Float),
                .regular => |decoded| blk: {
                    break :blk encodeFloat(Float, Fraction, Exponent, decoded.frac, decoded.exp, decoded.sign);
                },
            };
        }

        pub fn fromPosit(x: anytype) Self {
            const T = @TypeOf(x);
            return switch (x.decode()) {
                .zero => zero,
                .inf => inf,
                .regular => |v| encode(T.Fraction, T.Exponent, v.frac, v.exp, v.sign),
            };
        }

        pub fn fromInt(comptime T: type, x: T) Self {
            std.debug.assert(@typeInfo(T) == .int);
            if (x == 0) return zero;
            var abs_x = @abs(x);
            const exp = @bitSizeOf(@TypeOf(abs_x)) - 1 - @clz(abs_x);
            abs_x = @shlExact(abs_x, @intCast(@clz(abs_x)));
            abs_x <<= 1;
            return encode(@TypeOf(abs_x), @TypeOf(exp), abs_x, exp, if (x > 0) .positive else .negative);
        }

        pub fn fromFloat(comptime T: type, x: T) Self {
            std.debug.assert(@typeInfo(T) == .float);
            return switch (decodeFloat(T, x)) {
                .zero => zero,
                .inf => inf,
                .regular => |v| encode(@TypeOf(v.fraction), @TypeOf(v.exponent), v.fraction, v.exponent, v.sign),
            };
        }

        pub fn fromPow2(comptime T: type, exp: T) Self {
            return encode(u0, T, 0, exp, .positive);
        }

        pub fn add(self: Self, other: Self) Self {
            //TODO: fast path if equal or opposite
            const big, const small = if (self.abs().le(other.abs())) .{ other, self } else .{ self, other };

            if (big.eq(zero)) return small;
            if (small.eq(zero)) return big;
            if (big.eq(inf)) return inf;
            if (small.eq(inf)) return inf;
            const big_regular = big.decode().regular;
            const small_regular = small.decode().regular;

            const frac_size = @bitSizeOf(Fraction);
            const exp_size = @bitSizeOf(Exponent);
            const AddInt = Int(.unsigned, 2 * (frac_size + 1) + 2);
            const big_int = (1 << frac_size | @as(AddInt, big_regular.frac)) << (frac_size + 2);
            var small_int = (1 << frac_size | @as(AddInt, small_regular.frac)) << (frac_size + 2);

            const ExpAddInt = Int(.signed, exp_size + 2);
            const exp_difference = @as(ExpAddInt, big_regular.exp) - @as(ExpAddInt, small_regular.exp); // [0, 2*max_exponent]
            if (exp_difference > frac_size + 2) return big;

            const sht_amt: IntFittingRange(0, frac_size + 2) = @intCast(exp_difference);
            small_int = @shrExact(small_int, sht_amt);

            var result_int: AddInt = if (big_regular.sign == small_regular.sign) big_int + small_int else big_int - small_int;

            if (result_int == 0) return zero;
            const result_shf: IntFittingRange(0, frac_size + 1) = @intCast(@clz(result_int));
            result_int = @shlExact(result_int, result_shf);
            result_int <<= 1; //rimuovo leading 1
            const result_esp = big_regular.exp - @as(ExpAddInt, @intCast(result_shf)) + 1;

            return encode(AddInt, ExpAddInt, result_int, result_esp, big_regular.sign);
        }

        pub fn mul(self: Self, other: Self) Self {
            //TODO: fast path for one
            if (other.eq(inf)) return inf;
            if (self.eq(inf)) return inf;
            if (other.eq(zero)) return zero;
            if (self.eq(zero)) return zero;
            const self_regular = self.decode().regular;
            const other_regular = other.decode().regular;

            const frac_size = @bitSizeOf(Fraction);
            const MulInt = Int(.unsigned, frac_size + 1);
            const MulExpInt = Int(.signed, @bitSizeOf(Exponent) + 2);
            const self_int: MulInt = @as(MulInt, self_regular.frac) | 1 << frac_size;
            const other_int: MulInt = @as(MulInt, other_regular.frac) | 1 << frac_size;
            var exp = @as(MulExpInt, self_regular.exp) + @as(MulExpInt, other_regular.exp) + 1;

            var mul_int = std.math.mulWide(MulInt, self_int, other_int);
            const mul_shift_amt: u1 = @intCast(@clz(mul_int)); //clz, but only possible values 0,1
            mul_int = @shlExact(mul_int, mul_shift_amt);
            exp -= mul_shift_amt;
            mul_int <<= 1;
            const mul_sign: Sign = if (self_regular.sign == other_regular.sign) .positive else .negative;
            return encode(@TypeOf(mul_int), MulExpInt, mul_int, exp, mul_sign);
        }

        pub fn div(self: Self, other: Self) Self {
            //TODO: fast path for one

            if (other.eq(inf)) return inf;
            if (other.eq(zero)) return inf;
            if (self.eq(inf)) return inf;
            if (self.eq(zero)) return zero;
            const self_regular = self.decode().regular;
            const other_regular = other.decode().regular;

            const frac_size = @bitSizeOf(Fraction);
            const Numerator = Int(.unsigned, 2 * (frac_size + 1) + 1);
            const Denominator = Int(.unsigned, frac_size + 1);
            const ExpDiv = Int(.signed, @bitSizeOf(Exponent) + 2);
            var exp = @as(ExpDiv, self_regular.exp) - @as(ExpDiv, other_regular.exp);
            const num = (1 << frac_size | @as(Numerator, self_regular.frac)) << 2 + frac_size;
            const den = 1 << frac_size | @as(Denominator, other_regular.frac);
            var result_int = @divTrunc(num, den);
            const rem_int = @rem(num, den);
            const shf_amt = 1 - @as(u1, @intCast(@clz(result_int) - frac_size));
            result_int >>= shf_amt;
            exp -= 1 - shf_amt;
            var frac: Int(.unsigned, frac_size + 2) = @truncate(result_int << 1);
            frac += @intFromBool(rem_int != 0);

            const div_sign: Sign = if (self_regular.sign == other_regular.sign) .positive else .negative;
            return encode(@TypeOf(frac), ExpDiv, frac, exp, div_sign);
        }

        pub fn inv(self: Self) Self {
            return one.div(self);
        }

        pub fn sub(self: Self, other: Self) Self {
            return self.add(other.negate());
        }

        pub fn pinv(self: Self) Self {
            return Self{ .bits = inf.bits -% self.bits };
        }

        pub fn sqrt(self: Self) Self {
            return switch (self.decode()) {
                .inf => inf,
                .zero => zero,
                .regular => |self_regular| blk: {
                    if (self_regular.sign == .negative) break :blk inf;
                    const frac_size = @bitSizeOf(Fraction);
                    const SqrtInt = Int(.unsigned, 2 + 2 * (frac_size + 1));
                    const exp = self_regular.exp >> 1;
                    var frac_int = (1 << frac_size | @as(SqrtInt, self_regular.frac)) << frac_size + 2;
                    //frac_int <<= @truncate(u1, @bitCast(Int(.unsigned, @bitSizeOf(Exponent)), self_regular.exp));
                    frac_int <<= @as(u1, @truncate(@as(Int(.unsigned, @bitSizeOf(Exponent)), @bitCast(self_regular.exp))));
                    //const res = sqrt_int(SqrtInt, frac_int);
                    //const result_int = res.res;
                    //const carry = @boolToInt(res.carry!=0);
                    const result_int = rootNInt(SqrtInt, frac_int, 2);
                    const carry = @intFromBool(std.math.mulWide(@TypeOf(result_int), result_int, result_int) != frac_int);
                    const frac = (result_int << 1) + carry;
                    break :blk encode(@TypeOf(frac), Exponent, frac, exp, .positive);
                },
            };
        }

        pub fn rootN(self: Self, comptime n: comptime_int) Self {
            const self_regular = switch (self.decode()) {
                .inf => return inf,
                .zero => return zero,
                .regular => |regular| regular,
            };

            if (self_regular.sign == .negative and n & 1 == 0) return inf;
            const frac_size = @bitSizeOf(Fraction);
            const RootInt = Int(.unsigned, n + n * (frac_size + 1));
            const exp = @divFloor(self_regular.exp, n);

            const rem_exp: IntFittingRange(0, n - 1) = @intCast(self_regular.exp - exp * n);
            var frac_int = (1 << frac_size | @as(RootInt, self_regular.frac)) << ((frac_size + 1) * (n - 1) + 1);
            frac_int <<= rem_exp;
            const resultInt = rootNInt(RootInt, frac_int, n);

            var x: RootInt = 1;
            inline for (0..n) |_| x *= @as(RootInt, resultInt);
            const carry = @intFromBool(x != frac_int);
            const frac = (resultInt << 1) + carry;
            return encode(@TypeOf(frac), Exponent, frac, exp, self_regular.sign);
        }

        pub fn fMM(self: Self, other1: Self, other2: Self) Self {
            if (self.eq(inf)) return inf;
            if (other1.eq(inf)) return inf;
            if (other2.eq(inf)) return inf;
            if (self.eq(zero)) return zero;
            if (other1.eq(zero)) return zero;
            if (other2.eq(zero)) return zero;
            const self_regular = self.decode().regular;
            const other1_regular = other1.decode().regular;
            const other2_regular = other2.decode().regular;
            const frac_size = @bitSizeOf(Fraction);
            const MulInt = Int(.unsigned, (frac_size + 1) * 3);
            const MulEspInt = Int(.signed, @bitSizeOf(Exponent) + 2);
            const self_int: MulInt = @as(MulInt, self_regular.frac) | 1 << frac_size;
            const other1_int: MulInt = @as(MulInt, other1_regular.frac) | 1 << frac_size;
            const other2_int: MulInt = @as(MulInt, other2_regular.frac) | 1 << frac_size;
            var exp = @as(MulEspInt, self_regular.exp) + @as(MulEspInt, other1_regular.exp) + @as(MulEspInt, other2_regular.exp) + 2;

            var mul_int = self_int * other1_int * other2_int;
            const mul_shift_amt: u2 = @intCast(@clz(mul_int)); //clz, but only possible values 0,1,2
            mul_int = @shlExact(mul_int, mul_shift_amt);
            exp -= mul_shift_amt;
            mul_int <<= 1;
            var mul_sign: Sign = if (self_regular.sign == other1_regular.sign) .positive else .negative;
            mul_sign = if (mul_sign == other2_regular.sign) .positive else .negative;
            return encode(MulInt, MulEspInt, mul_int, exp, mul_sign);
        }

        pub fn mod(self: Self, other: Self) Self {
            return self.sub(self.div(other).floor().mul(other));
        }

        pub fn rem(self: Self, other: Self) Self {
            return self.sub(self.div(other).floorToZero().mul(other));
        }

        pub fn toQuire(self: Self) Quire {
            return Quire.fromPosit(self);
        }

        pub const Quire = packed struct {
            pub const quire_len: comptime_int = std.math.ceilPowerOfTwoPromote(usize, (4 * nbit - 8) * (1 << es) + 1 + 30);
            pub const max_quire_exp = quire_len - 1 - quire_frac_size;
            pub const min_quire_exp = -quire_frac_size;
            pub const QuireExponent = IntFittingRange(min_quire_exp, max_quire_exp);
            pub const quire_frac_size = max_exponent * 2;
            pub const UIntQuire = Int(.unsigned, quire_len);
            pub const IntQuire = Int(.signed, quire_len);

            bits: UIntQuire,

            pub const zero_quire = Quire{ .bits = 0 };
            pub const inf_quire = Quire{ .bits = 1 << (quire_len - 1) };
            pub const one_quire = Quire{ .bits = 1 << quire_frac_size };
            pub const min_quire = zero_quire.next();
            pub const max_quire = inf_quire.prior();

            pub fn isNegative(self: Quire) bool {
                return @as(IntQuire, @bitCast(self.bits)) < 0;
            }
            pub fn negate(self: Quire) Quire {
                return .{ .bits = @bitCast(-%@as(IntQuire, @bitCast(self.bits))) };
            }
            pub fn abs(self: Quire) Quire {
                return if (self.isNegative()) self.negate() else self;
            }
            pub fn next(self: Quire) Quire {
                return .{ .bits = self.bits +% 1 };
            }
            pub fn prior(self: Quire) Quire {
                return .{ .bits = self.bits -% 1 };
            }
            pub fn le(self: Quire, other: Quire) bool {
                return @as(IntQuire, @bitCast(self.bits)) < @as(IntQuire, @bitCast(other.bits));
            }
            pub fn gr(self: Quire, other: Quire) bool {
                return @as(IntQuire, @bitCast(self.bits)) > @as(IntQuire, @bitCast(other.bits));
            }
            pub fn leEq(self: Quire, other: Quire) bool {
                return @as(IntQuire, @bitCast(self.bits)) <= @as(IntQuire, @bitCast(other.bits));
            }
            pub fn grEq(self: Quire, other: Quire) bool {
                return @as(IntQuire, @bitCast(self.bits)) >= @as(IntQuire, @bitCast(other.bits));
            }
            pub fn eq(self: Quire, other: Quire) bool {
                return self.bits == other.bits;
            }

            pub fn decode(self: Quire) union(enum) {
                zero,
                inf,
                regular: struct {
                    frac: UIntQuire,
                    exp: QuireExponent,
                    sign: Sign,
                },
            } {
                if (self.bits == inf_quire.bits) return .inf;
                if (self.bits == zero_quire.bits) return .zero;
                const sign_out: Sign = if (self.isNegative()) .negative else .positive;
                var frac = self.abs().bits;
                const shf_amt = @clz(frac);
                frac = @shlExact(frac, @intCast(shf_amt));
                frac <<= 1;
                var exp: QuireExponent = max_quire_exp;
                exp -= @intCast(shf_amt);
                //const exp = max_quire_exp - @intCast(QuireExponent, shf_amt);
                return .{ .regular = .{ .frac = frac, .exp = exp, .sign = sign_out } };
            }
            //doesn't correctly approximate
            pub fn encode(comptime T: type, comptime M: type, frac: T, exp: M, outSign: Sign) Quire {
                std.debug.assert(@typeInfo(T) == .int);
                std.debug.assert(@typeInfo(T).int.signedness == .unsigned);
                std.debug.assert(@typeInfo(M) == .int);
                if (exp >= max_quire_exp) return if (outSign == .positive) max_quire else max_quire.negate();
                if (exp < min_quire_exp) return if (outSign == .positive) min_quire else min_quire.negate();
                const quire_exp: QuireExponent = @intCast(exp);
                const frac_size = @bitSizeOf(T);
                const shf_amt = quire_frac_size - frac_size;
                var val: UIntQuire = undefined;
                if (shf_amt < 0) {
                    const ShiftInt = IntFittingRange(min_quire_exp + @min(0, shf_amt), max_quire_exp + @max(0, shf_amt));
                    const remaining_frac = std.math.shl(T, frac, @as(ShiftInt, quire_exp) + shf_amt);
                    val = @intCast(remaining_frac);
                    val |= std.math.shl(UIntQuire, one_quire.bits, quire_exp);
                } else {
                    val = one_quire.bits | @as(UIntQuire, frac) << shf_amt;
                    val = std.math.shl(UIntQuire, val, quire_exp);
                }
                var out = Quire{ .bits = val };
                if (outSign == .negative) out = out.negate();
                return out;
            }

            pub fn toFloat(self: Quire, comptime Float: type) Float {
                std.debug.assert(@typeInfo(Float) == .float);
                return switch (self.decode()) {
                    .zero => 0,
                    .inf => std.math.nan(Float),
                    .regular => |r| encodeFloat(Float, UIntQuire, QuireExponent, r.frac, r.exp, r.sign),
                };
            }

            pub fn toPosit(self: Quire) Self {
                return switch (self.decode()) {
                    .zero => Self.zero,
                    .inf => Self.inf,
                    .regular => |r| Self.encode(UIntQuire, QuireExponent, r.frac, r.exp, r.sign),
                };
            }

            pub fn toPositSplit(self: Quire) struct { Self, Self } {
                return switch (self.decode()) {
                    .zero => .{ Self.zero, Self.zero },
                    .inf => .{ Self.zero, Self.zero },
                    .regular => |r| .{
                        Self.encode(u0, QuireExponent, 0, r.exp, r.sign),
                        Self.encode(UIntQuire, u0, r.frac, 0, .positive),
                    },
                };
            }

            pub fn fromInt(comptime T: type, x: T) Quire {
                std.debug.assert(@typeInfo(T) == .int);
                if (x == 0) return zero;
                var abs_x = @abs(x);
                const exp = @bitSizeOf(@TypeOf(abs_x)) - 1 - @clz(abs_x);
                abs_x = @shlExact(abs_x, @intCast(@clz(abs_x)));
                abs_x <<= 1;
                return Quire.encode(@TypeOf(abs_x), @TypeOf(exp), abs_x, exp, if (x > 0) .positive else .negative);
            }

            pub fn fromFloat(comptime T: type, x: T) Quire {
                std.debug.assert(@typeInfo(T) == .float);
                return switch (decodeFloat(T, x)) {
                    .zero => zero_quire,
                    .inf => inf_quire,
                    .regular => |v| Quire.encode(@TypeOf(v.fraction), @TypeOf(v.exponent), v.fraction, v.exponent, v.sign),
                };
            }

            pub fn fromPosit(posit: Self) Quire {
                return switch (posit.decode()) {
                    .inf => inf_quire,
                    .zero => zero_quire,
                    .regular => |r| Quire.encode(Fraction, Exponent, r.frac, r.exp, r.sign),
                };
            }

            pub fn add(self: Quire, other: Quire) Quire {
                if (self.bits == inf_quire.bits) return inf_quire;
                if (other.bits == inf_quire.bits) return inf_quire;
                const self_int: IntQuire = @bitCast(self.bits);
                const other_int: IntQuire = @bitCast(other.bits);
                var res = self_int +| other_int;
                if (@as(UIntQuire, @bitCast(res)) == inf_quire.bits) res += 1;
                return Quire{ .bits = @bitCast(res) };
            }

            pub fn sub(self: Quire, other: Quire) Quire {
                return self.add(other.negate());
            }

            pub fn mul(posit1: Self, posit2: Self) Quire {
                if (posit1.eq(inf)) return inf_quire;
                if (posit2.eq(inf)) return inf_quire;
                if (posit1.eq(zero)) return zero_quire;
                if (posit2.eq(zero)) return zero_quire;

                const posit1_regular = posit1.decode().regular;
                const posit2_regular = posit2.decode().regular;

                const frac_bits = @bitSizeOf(Fraction);
                const MulInt = Int(.unsigned, frac_bits + 1);
                const MulExpInt = Int(.signed, @bitSizeOf(Exponent) + 2);
                const posit1Int: MulInt = @as(MulInt, posit1_regular.frac) | 1 << frac_bits;
                const posit2Int: MulInt = @as(MulInt, posit2_regular.frac) | 1 << frac_bits;
                var exp = @as(MulExpInt, posit1_regular.exp) + @as(MulExpInt, posit2_regular.exp) + 1;

                var mul_int = std.math.mulWide(MulInt, posit1Int, posit2Int);
                const mul_shift_amt: u1 = @intCast(@clz(mul_int)); //clz, but only possible values 0,1
                mul_int = @shlExact(mul_int, mul_shift_amt);
                exp -= mul_shift_amt;
                mul_int <<= 1;
                const mul_sign: Sign = if (posit1_regular.sign == posit2_regular.sign) .positive else .negative;
                return Quire.encode(@TypeOf(mul_int), MulExpInt, mul_int, exp, mul_sign);
            }

            pub fn mulAdd(self: Quire, posit1: Self, posit2: Self) Quire {
                return self.add(Quire.mul(posit1, posit2));
            }
            pub fn mulSub(self: Quire, posit1: Self, posit2: Self) Quire {
                return self.sub(Quire.mul(posit1, posit2));
            }
            pub fn sqrt(self: Quire) Quire {
                if (self.isNegative()) return inf_quire;
                const SqrtInt = Int(.unsigned, quire_len + quire_frac_size);
                const sqrt_int = @as(SqrtInt, self.bits) << quire_frac_size;
                const result_int = std.math.sqrt(sqrt_int);
                return Quire{ .bits = result_int };
            }

            pub fn format(value: Quire, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
                const f = value.toFloat(f64);
                _ = options;
                return std.fmt.format(writer, "{" ++ fmt ++ "}", .{f});
            }
        };
        pub fn format(value: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            if (comptime std.mem.startsWith(u8, fmt, "b")) {
                return std.fmt.format(writer, "{" ++ fmt ++ "}", .{value.bits});
            } else {
                const f = value.toFloat(f64);
                _ = options;
                return std.fmt.format(writer, "{" ++ fmt ++ "}", .{f});
            }
        }
    };
}

fn IntCeil(comptime T: type) type {
    const bitSizes = [_]comptime_int{ 8, 16, 32, 64, 128 };
    const size = @bitSizeOf(T);
    const signedness = @typeInfo(T).int.signedness;
    for (bitSizes) |b| {
        if (b >= size) return std.meta.int(signedness, b);
    }
    return T;
}

fn rootNInt(comptime T: type, value: T, comptime n: comptime_int) RootN(T, n) {
    var res: T = 0;
    if (value == 0) return 0;
    if (value < 1 << n) return 1;
    const bits = @bitSizeOf(T);
    var j: T = 1 << @divTrunc(bits - 1, n) * n;
    var one: T = 1 << @divTrunc(bits - 1, n);
    while (value < j) : (j >>= n) {
        one >>= 1;
    }
    while (one != 0) : (one >>= 1) {
        const cand = res + one;
        var x1: Int(.unsigned, @divTrunc(bits - 1, n) * n + n) = 1;
        inline for (0..n) |_| x1 *= cand;
        if (x1 <= value) res = cand;
    }
    return @intCast(res);
}

fn RootN(comptime T: type, comptime n: T) type {
    return std.meta.Int(.unsigned, (@bitSizeOf(T) + n - 1) / n);
}


test "1-1 = 0" {
    const bit_sizes = [_]usize{ 8, 32, 32, 64 };
    const es_sizes = [_]usize{ 2, 3, 4 };
    inline for (bit_sizes) |bitSize| {
        inline for (es_sizes) |esSize| {
            const PositType = Posit(bitSize, esSize);
            const one = PositType.one;
            const zero = PositType.zero;
            const sub = one.sub(one);
            try std.testing.expect(sub.eq(zero));
        }
    }
}

test "conversions" {
    const Posit8 = Posit(8, 2);
    const Posit16 = Posit(16, 2);
    
    const a8: Posit8 = .fromFloat(f64, 3);
    const a16: Posit16 = .fromFloat(f64, 3);
    const b16: Posit16 = .fromPosit(a8);
    
    try std.testing.expect(b16.eq(a16));
}
