const std = @import("std");
const ArrayList = std.ArrayList;

const Ops = enum { add, mul };

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const Value = struct {
    data: f32,
    grad: f32 = 0.0,
    prev: ?[2]?*Value,
    op: ?Ops,

    pub fn init(data: f32) !Value {
        var children = try allocator.create([2]?*Value);
        children[0] = null;
        children[1] = null;
        return Value{ .data = data, .prev = children.*, .op = undefined };
    }

    pub fn add(self: *Value, other: *Value) Value {
        var out = Value.init(self.data + other.data) catch |err| {
            std.debug.print("add err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = undefined };
        };
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.add;
        return out;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * other.data) catch |err| {
            std.debug.print("mul err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = undefined };
        };
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.mul;
        return out;
    }

    pub fn print(self: *Value) void {
        std.debug.print("data: {}\n", .{self.data});
        std.debug.print("op: {}\n", .{self.op.?});
    }

    pub fn print_prev(self: *Value) void {
        if (self.prev != null) {
            if (self.prev.?[0] != null)
                self.prev.?[0].?.print();
            if (self.prev.?[1] != null)
                self.prev.?[1].?.print();
        }
    }
};

test "junk" {
    var a = try Value.init(3.0);
    var b = try Value.init(2.0);

    var c = a.add(&b);
    c.print_prev();
}

test "simple add test" {
    var a = try Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.add(&b);
    try std.testing.expect(c.data == 5.0);
}

test "simple add test one neg" {
    var a = try Value.init(2.0);
    try std.testing.expect(a.data == 2.0);
    var b = try Value.init(-3.0);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.data == -1.0);
}

test "simple add test two neg" {
    var a = try Value.init(-2.0);
    try std.testing.expect(a.data == -2.0);
    var b = try Value.init(-3.0);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.data == -5.0);
}

test "simple mul test" {
    var a = try Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.data == 6.0);
}

test "simple mul test one neg" {
    var a = try Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.data == -6.0);
}

test "simple mul test two neg" {
    var a = try Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = try Value.init(-2.0);
    try std.testing.expect(b.data == -2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.data == 6.0);
}
