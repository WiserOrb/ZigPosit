# ZigPosit

A pure zig posit implementation for generic bitlen and es values, even odd ones

```zig
const Posit = @import("zigposit").Posit;

// Type definitions
const Posit8 = Posit(8, 2);
const Posit32 = Posit(32, 2);
const Posit17_5 = Posit(17, 5);

// Inizialization
const a = Posit8.fromFloat(f32, 10.34);
const b = Posit32.fromInt(i32, -42);
const c = Posit17_5.fromPosit(Posit32, b);

// aritmetics
_ = a.add(a)
... sub, mul, div, sqrt,sqrtN, fMM, 


// Quire
const q_a = a.toQuire();
const q_b = a.mulAdd(a, b.toPosit(Posit8));


```

Tested against softposit implementation
https://gitlab.com/cerlane/SoftPosit
