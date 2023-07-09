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
        // TODO: Look into another way of allocating and deinit function for freeing memory
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

    fn backwardAdd(self: *Value) void {
        self.prev.?[0].?.grad += self.grad;
        self.prev.?[1].?.grad += self.grad;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * other.data);
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.mul;
        return out;
    }

    fn backwardMul(self: *Value) void {
        self.prev.?[0].?.grad += self.prev.?[1].?.data * self.grad;
        self.prev.?[1].?.grad += self.prev.?[0].?.data * self.grad;
    }

    pub fn pow(self: *Value, other: *Value) Value {
        var out = Value.init(std.math.pow(f32, self.data, other.data));
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.pow;
        return out;
    }

    fn backwardPow(self: *Value) void {
        self.prev.?[0].?.grad += self.prev.?[1].?.data * std.math.pow(f32, self.prev.?[0].?.data, self.prev.?[1].?.data) * self.grad;
    }

    pub fn relu(self: *Value) Value {
        var out = if (self.data < 0) Value.init(0.0) else Value.init(self.data);
        out.prev.?[0] = self;
        out.op = Ops.relu;
        return out;
    }

    fn backwardRelu(self: *Value) void {
        self.prev.?[0].?.grad += if (self.data > 0.0) self.grad else 0;
    }

    pub fn neg(self: *Value) Value {
        var out = Value.init(self.data * -1);
        out.prev.?[0] = self;
        out.op = Ops.neg;
        return out;
    }

    fn backwardNeg(self: *Value) void {
        self.prev.?[0].?.grad -= self.grad;
    }

    pub fn sub(self: *Value, other: *Value) Value {
        var out = Value.init(self.data - other.data);
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.sub;
        return out;
    }

    fn backwardSub(self: *Value) void {
        self.prev.?[0].?.grad += self.grad;
        self.prev.?[1].?.grad -= self.grad;
    }

    pub fn div(self: *Value, other: *Value) Value {
        var out = Value.init(self.data * std.math.pow(f32, other.data, -1));
        out.prev.?[0] = self;
        out.prev.?[1] = other;
        out.op = Ops.div;
        return out;
    }

    fn backwardDiv(self: *Value) void {
        const b: f32 = self.prev.?[1].?.data;
        self.prev.?[0].?.grad += (1.0 / b) * self.grad;
        self.prev.?[1].?.grad -= (self.prev.?[0].?.data / (b * b)) * self.grad;
    }

    pub fn print(self: *Value) void {
        std.debug.print("data: {}\n", .{self.data});
        std.debug.print("op: {?}\n", .{self.op});
        std.debug.print("grad: {}\n", .{self.grad});
    }

    pub fn printPrev(self: *Value) void {
        if (self.prev != null) {
            if (self.prev.?[0] != null)
                self.prev.?[0].?.print();
            if (self.prev.?[1] != null)
                self.prev.?[1].?.print();
        }
    }

    pub fn backward(self: *Value) void {
        var topo = ArrayList(Value).init(allocator);
        var visited = ArrayList(Value).init(allocator);
        buildTopo(&topo, &visited, self.*);
        self.grad = 1.0;

        var i = topo.items.len;
        while (i > 0) : (i -= 1) {
            if (self.op != null) {
                switch (self.op.?) {
                    Ops.add => {
                        self.backwardAdd();
                    },
                    Ops.sub => {
                        self.backwardSub();
                    },
                    Ops.mul => {
                        self.backwardMul();
                    },
                    Ops.div => {
                        self.backwardMul();
                    },
                    Ops.pow => {
                        self.backwardPow();
                    },
                    Ops.neg => {
                        self.backwardNeg();
                    },
                    Ops.relu => {
                        self.backwardNeg();
                    },
                }
            }
        }
    }

    pub fn buildTopo(topo: *ArrayList(Value), visited: *ArrayList(Value), value: Value) void {
        if (!contains(visited.*, value)) {
            visited.*.append(value) catch |err| {
                std.debug.panic("Error when appending to visited list in buildTopo: {}", .{err});
            };
            if (value.prev != null) {
                for (value.prev.?) |child| {
                    if (child != null) {
                        buildTopo(topo, visited, child.?.*);
                    }
                }
            }
            topo.append(value) catch |err| {
                std.debug.panic("Error when appending to topo list in buildTopo: {}", .{err});
            };
        }
    }
};

fn contains(list: ArrayList(Value), val: Value) bool {
    var exists = false;
    for (list.items) |value| {
        // TODO: This is not ideal, this will not follow pointers
        if (std.meta.eql(val, value)) {
            exists = true;
        }
    }
    return exists;
}

const expect = std.testing.expect;

test "backward" {
    var a = Value.init(3.0);
    var b = Value.init(4.0);
    var c = a.add(&b);
    c.backward();
    c.print();
    c.printPrev();
}

test "contains test" {
    var someList = ArrayList(Value).init(std.testing.allocator);
    defer someList.deinit();
    var a = Value.init(2.0);
    var b = Value.init(3.0);
    var c = a.add(&b);
    var d = Value.init(4.0);
    var e = c.mul(&d);
    try someList.append(a);
    try someList.append(b);
    try someList.append(c);
    try someList.append(e);

    try expect(contains(someList, a));
    try expect(contains(someList, b));
    try expect(contains(someList, c));
    try expect(!contains(someList, d));
    try expect(contains(someList, e));
}

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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 2.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == -2.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 3.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
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

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == -3.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == -2.0);
}

test "simple pow test" {
    var a = Value.init(2.0);
    var b = Value.init(2.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 4.0);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 2.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple pow test neg" {
    var a = Value.init(2.0);
    var b = Value.init(-1.0);

    var c = a.pow(&b);

    try std.testing.expect(c.op == Ops.pow);
    try std.testing.expect(c.data == 0.5);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 2.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == -1.0);
}

test "simple relu test larger than 0" {
    var a = Value.init(2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 2.0);

    try std.testing.expect(b.prev.?[0].?.op == null);
    try std.testing.expect(b.prev.?[0].?.data == 2.0);

    try std.testing.expect(b.prev.?[1] == null);
}

test "simple relu test less than 0" {
    var a = Value.init(-2.0);
    try std.testing.expect(a.op == null);
    try std.testing.expect(a.data == -2.0);
    var b = a.relu();

    try std.testing.expect(b.op == Ops.relu);
    try std.testing.expect(b.data == 0.0);

    try std.testing.expect(b.prev.?[0].?.op == null);
    try std.testing.expect(b.prev.?[0].?.data == -2.0);

    try std.testing.expect(b.prev.?[1] == null);
}

test "simple neg test" {
    var a = Value.init(3.0);
    try std.testing.expect(a.data == 3.0);
    var b = a.neg();
    try std.testing.expect(b.op == Ops.neg);
    try std.testing.expect(b.data == -3.0);

    try std.testing.expect(b.prev.?[0].?.op == null);
    try std.testing.expect(b.prev.?[0].?.data == 3.0);

    try std.testing.expect(b.prev.?[1] == null);
}

test "simple double neg test" {
    var a = Value.init(-3.0);
    try std.testing.expect(a.data == -3.0);
    var b = a.neg();
    try std.testing.expect(b.op == Ops.neg);
    try std.testing.expect(b.data == 3.0);

    try std.testing.expect(b.prev.?[0].?.op == null);
    try std.testing.expect(b.prev.?[0].?.data == -3.0);

    try std.testing.expect(b.prev.?[1] == null);
}

test "simple sub test" {
    var a = Value.init(14.0);
    var b = Value.init(7.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == 7.0);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 14.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == 7.0);
}

test "simple sub neg test" {
    var a = Value.init(7.0);
    var b = Value.init(14.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == -7.0);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 7.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == 14.0);
}

test "simple sub double neg test" {
    var a = Value.init(14.0);
    var b = Value.init(-7.0);
    var c = a.sub(&b);
    try std.testing.expect(c.op == Ops.sub);
    try std.testing.expect(c.data == 21.0);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 14.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == -7.0);
}

test "simple div test" {
    var a = Value.init(5.0);
    var b = Value.init(2.0);
    var c = a.div(&b);
    try std.testing.expect(c.data == 2.5);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 5.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == 2.0);
}

test "simple div neg test" {
    var a = Value.init(5.0);
    var b = Value.init(-2.0);
    var c = a.div(&b);
    try std.testing.expect(c.data == -2.5);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 5.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == -2.0);
}

test "simple div less than 0 test" {
    var a = Value.init(5.0);
    var b = Value.init(0.5);
    var c = a.div(&b);
    try std.testing.expect(c.data == 10);

    try std.testing.expect(c.prev.?[0].?.op == null);
    try std.testing.expect(c.prev.?[0].?.data == 5.0);

    try std.testing.expect(c.prev.?[1].?.op == null);
    try std.testing.expect(c.prev.?[1].?.data == 0.5);
}

test "long multi op test" {
    var a = Value.init(5.0);
    var b = Value.init(0.5);
    var c = a.add(&b);
    var d = a.add(&b);
    var e = d.mul(&c);
    try expect(e.data == 30.25);
}
