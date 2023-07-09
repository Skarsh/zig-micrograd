const std = @import("std");
const ArrayList = std.ArrayList;

const Ops = enum { add, mul, pow, relu, neg, sub, div };

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const Value = struct {
    data: f32,
    grad: f32 = 0.0,
    prev: ?[2]?*Value,
    op: ?Ops,

    pub fn init(data: f32) Value {
        var children = allocator.create([2]?*Value) catch |err| {
            std.debug.panic("Error: {}", .{err});
        };
        children[0] = null;
        children[1] = null;
        return Value{ .data = data, .prev = children.*, .op = null };
    }

    pub fn add(self: *Value, other: *Value) Value {
        var out = Value.init(self.data + other.data);
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.add;
        return out;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * other.data);
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.mul;
        return out;
    }

    pub fn pow(self: *Value, other: *Value) Value {
        var out = Value.init(std.math.pow(f32, self.data, other.data));
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.pow;
        return out;
    }

    pub fn relu(self: *Value) Value {
        var out = if (self.data < 0) Value.init(0.0) else Value.init(self.data);
        out.prev.?[0] = self;
        out.op = Ops.relu;
        return out;
    }

    pub fn neg(self: *Value) Value {
        var out = Value.init(self.data * -1);
        out.prev.?[0] = self;
        out.op = Ops.neg;
        return out;
    }

    pub fn sub(self: *Value, other: *Value) Value {
        var out = Value.init(self.data - other.data);
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.sub;
        return out;
    }

    pub fn div(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * std.math.pow(f32, other.data, -1));
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.div;
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

const expect = std.testing.expect;

test "simple add test" {
    var a = Value.init(3.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == 3.0);
    var b = Value.init(2.0);
    try std.testing.expect(b.op == null);
    try std.testing.expect(b.data == 2.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == 5.0);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple add test one neg" {
    var a = Value.init(2.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == 2.0);
    var b = Value.init(-3.0);
    try std.testing.expect(b.op == null);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == -1.0);
    try std.testing.expect(c.prev.?[0].?.data == 2.0);
    try std.testing.expect(c.prev.?[1].?.data == -3.0);
}

test "simple add test two neg" {
    var a = Value.init(-2.0);
    try std.testing.expect(a.data == -2.0);
    var b = Value.init(-3.0);
    try std.testing.expect(b.data == -3.0);

    var c = a.add(&b);
    try std.testing.expect(c.op == Ops.add);
    try std.testing.expect(c.data == -5.0);
    try std.testing.expect(c.prev.?[0].?.data == -2.0);
    try std.testing.expect(c.prev.?[1].?.data == -3.0);
}

test "simple mul test" {
    var a = Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == 6.0);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple mul test one neg" {
    var a = Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = Value.init(2.0);
    try std.testing.expect(b.data == 2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == -6.0);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple mul test two neg" {
    var a = Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = Value.init(-2.0);
    try std.testing.expect(b.data == -2.0);

    var c = a.mul(&b);
    try std.testing.expect(c.op == Ops.mul);
    try std.testing.expect(c.data == 6.0);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);
    try std.testing.expect(c.prev.?[1].?.data == -2.0);
}

test "simple pow test" {
    var a = Value.init(2.0);
    var b = Value.init(2.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 4.0);
}

test "simple pow test neg" {
    var a = Value.init(2.0);
    var b = Value.init(-1.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 0.5);
}

test "simple relu test larger than 0" {
    var a = Value.init(2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 2.0);
}

test "simple relu test less than 0" {
    var a = Value.init(-2.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == -2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 0.0);
}

test "simple neg test" {
    var a = Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = a.neg();
    try std.testing.expect(b.op == Ops.neg);
    try std.testing.expect(b.data == -3.0);
}

test "simple double neg test" {
    var a = Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = a.neg();
    try std.testing.expect(b.op == Ops.neg);
    try std.testing.expect(b.data == 3.0);
}

test "simple sub test" {
    var a = Value.init(14.0);
    var b = Value.init(7.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == 7.0);
}

test "simple sub neg test" {
    var a = Value.init(7.0);
    var b = Value.init(14.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == -7.0);
}

test "simple sub double neg test" {
    var a = Value.init(14.0);
    var b = Value.init(-7.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == 21.0);
}

test "simple div test" {
    var a = Value.init(5.0);
    var b = Value.init(2.0);
    var c = a.div(&b);
    try std.testing.expect(c.data == 2.5);
}

test "simple div neg test" {
    var a = Value.init(5.0);
    var b = Value.init(-2.0);
    var c = a.div(&b);
    try std.testing.expect(c.data == -2.5);
}

test "simple div less than 0 test" {
    var a = Value.init(5.0);
    var b = Value.init(0.5);
    var c = a.div(&b);
    try std.testing.expect(c.data == 10);
}
