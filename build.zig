const std = @import("std");


pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const comath_mod = b.addModule("zigposit", .{
        .root_source_file = b.path("src/posit.zig"),
    });
}
