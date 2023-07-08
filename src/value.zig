const std = @import("std");
const ArrayList = std.ArrayList;

const Ops = enum { add, mul, pow, relu };

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
        return Value{ .data = data, .prev = children.*, .op = null };
    }

    pub fn add(self: *Value, other: *Value) Value {
        var out = Value.init(self.data + other.data) catch |err| {
            std.debug.print("add err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = null };
        };
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.add;
        return out;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * other.data) catch |err| {
            std.debug.print("mul err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = null };
        };
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.mul;
        return out;
    }

    pub fn pow(self: *Value, other: *Value) Value {
        var out = Value.init(std.math.pow(f32, self.data, other.data)) catch |err| {
            std.debug.print("mul err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = null };
        };
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.pow;
        return out;
    }

    pub fn relu(self: *Value) Value {
        var out = if (self.data < 0) Value.init(0.0) catch |err| {
            std.debug.print("mul err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = null };
        } else Value.init(self.data) catch |err| {
            std.debug.print("mul err {}", .{err});
            return Value{ .data = 0.0, .prev = null, .op = null };
        };
        out.prev.?[0] = self;
        out.op = Ops.relu;
        return out;
    }

    pub fn print(self: *Value) void {
        std.debug.print("data: {}\n", .{self.data});
        std.debug.print("op: {?}\n", .{self.op});
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

test "simple add test" {
    var a = try Value.init(3.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == 3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.op == null);
    try std.testing.expect(b.data == 2.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == 5.0);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple add test one neg" {
    var a = try Value.init(2.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == 2.0);
    var b = try Value.init(-3.0);
    try std.testing.expect(b.op == null);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == -1.0);
    try std.testing.expect(c.prev.?[0].?.data == 2.0);
    try std.testing.expect(c.prev.?[1].?.data == -3.0);
}

test "simple add test two neg" {
    var a = try Value.init(-2.0);
    try std.testing.expect(a.data == -2.0);
    var b = try Value.init(-3.0);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == -5.0);
    try std.testing.expect(c.prev.?[0].?.data == -2.0);
    try std.testing.expect(c.prev.?[1].?.data == -3.0);
}

test "simple mul test" {
    var a = try Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == 6.0);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple mul test one neg" {
    var a = try Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = try Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == -6.0);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple mul test two neg" {
    var a = try Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = try Value.init(-2.0);
    try std.testing.expect(b.data == -2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == 6.0);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);
    try std.testing.expect(c.prev.?[1].?.data == -2.0);
}

test "simple pow test" {
    var a = try Value.init(2.0);
    var b = try Value.init(2.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 4.0);
}

test "simple pow test neg" {
    var a = try Value.init(2.0);
    var b = try Value.init(-1.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 0.5);
}

test "simple relu test larger than 0" {
    var a = try Value.init(2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 2.0);
}

test "simple relu test less than 0" {
    var a = try Value.init(-2.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == -2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 0.0);
}
