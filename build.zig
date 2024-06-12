const std = @import("std");


pub fn build(b: *std.Build) void {
    _ = b.standardTargetOptions(.{});
   _ = b.standardOptimizeOption(.{});

    _ = b.addModule("zigposit", .{
        .root_source_file = b.path("src/posit.zig"),
    });
}
