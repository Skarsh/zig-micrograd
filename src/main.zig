const std = @import("std");
const c = @cImport({
    @cInclude("SDL.h");
});

const ArrayList = std.ArrayList;

const Ops = enum { add, mul };

const Value = struct {
    data: f32,
    grad: f32 = 0.0,
    prev: ArrayList(Value),
    op: ?Ops,

    pub fn init(data: f32, children: ArrayList(Value)) Value {
        return Value{ .data = data, .prev = children, .op = undefined };
    }

    pub fn add(self: *Value, other: Value) *Value {
        self.data += other.data;
        self.op = Ops.add;
        self.prev.append(self.*) catch |err| {
            std.debug.print("add err {}", .{err});
        };
        self.prev.append(other) catch |err| {
            std.debug.print("add err {}", .{err});
        };
        return self;
    }

    pub fn mul(self: *Value, other: Value) *Value {
        self.data *= other.data;
        self.op = Ops.mul;
        self.prev.append(self.*) catch |err| {
            std.debug.print("mul err {}", .{err});
        };
        self.prev.append(other) catch |err| {
            std.debug.print("mul err {}", .{err});
        };
        return self;
    }

    pub fn print(self: *Value) void {
        std.debug.print("data: {}\n", .{self.data});
    }

    pub fn print_prev(self: *Value) void {
        for (self.prev.items) |child| {
            std.debug.print("data: {}\n", .{child.data});
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = Value.init(2.0, ArrayList(Value).init(allocator));
    defer a.prev.deinit();
    var b = Value.init(-3.0, ArrayList(Value).init(allocator));
    defer b.prev.deinit();
    var e = Value.init(10.0, ArrayList(Value).init(allocator));
    defer b.prev.deinit();

    var d = a.mul(b).add(e);

    d.print();
    std.debug.print("\n", .{});
    d.print_prev();
}

const expect = std.testing.expect;

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
