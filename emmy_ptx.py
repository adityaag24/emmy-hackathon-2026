"""A direct, scalar f32 Kernel IR to PTX backend. No CUDA source is generated."""

from __future__ import annotations

import json
import math
import operator
import re
import struct
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Smem, Sync,
    Tile, TreeHalve, WarpShuffle,
)
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write

REGISTER_PREFIX = {"s32": "r", "f32": "f", "u64": "rd", "pred": "p"}
MOVE_TYPES = {"s32": "b32", "f32": "f32", "u64": "b64", "pred": "pred"}
COMPARISONS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
INTEGER_OPS = {"+": "add", "-": "sub", "*": "mul.lo", "^": "xor", "&": "and", "|": "or", ">>": "shr", "<<": "shl"}
FLOAT_OPS = {"*": "mul.rn", "+": "add.rn", "-": "sub.rn", "/": "div.rn"}
ELEMENTWISE_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
INTEGER_FOLD = {"+": operator.add, "-": operator.sub, "*": operator.mul,
                "/": operator.floordiv, "//": operator.floordiv, "%": operator.mod,
                "&": operator.and_, "|": operator.or_, "^": operator.xor}


class CompileError(ValueError):
    """An input requires a feature this backend cannot lower faithfully."""


def symbol(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name, flags=re.ASCII):
        raise CompileError(f"invalid PTX symbol {name!r}")
    return name


def walk(body):
    for stmt in body:
        yield stmt
        for child in stmt.nested():
            yield from walk(child)


@dataclass(frozen=True, slots=True)
class Value:
    text: str
    dtype: str
    integer: int | None = None
    minimum: int | None = None
    maximum: int | None = None


def constant(value, dtype="f32") -> Value:
    if dtype == "f32":
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return encoded_constant(bits, dtype)
    number = int(value)
    if not -(1 << 31) <= number < (1 << 31):
        raise CompileError("integer literal exceeds signed 32-bit indexing")
    return encoded_constant(number, dtype)


@lru_cache(maxsize=1024)
def encoded_constant(number, dtype):
    # Cache bit patterns, since Python considers +0.0 and -0.0 equal as keys.
    if dtype == "f32":
        return Value(f"0f{number:08X}", dtype)
    return Value(str(number), dtype, number, number, number)


class Compiler:
    def __init__(self, graph: Graph, node, launch: dict):
        self.kernel = node.op
        self.launch = launch
        self.lines: list[str] = []
        self.counts: Counter = Counter()
        self.labels = 0
        self.env: dict[str, Value] = {}
        self.buffers = {}
        self.pointers = {}
        self.shared = {}
        self.declarations = []
        self.module_declarations = []
        for n in graph.nodes.values():
            for tensor in n.outputs:
                self.buffers[tensor.name] = (tuple(d.as_static() for d in tensor.shape), tensor.dtype.name)
        if self.kernel.serial:
            raise CompileError("serial runtime axes are not supported")
        if launch["block"][1:] != [1, 1] or launch["grid"][1:] != [1, 1]:
            raise CompileError("only the Kernel IR's one dimensional launch is supported")
        self.params = launch["arg_order"]
        if len(set(self.params)) != len(self.params):
            raise CompileError("duplicate names in arg_order")
        args = {arg["name"]: arg for arg in launch["args"]}
        if set(args) != set(self.params) or not set(node.inputs).issubset(args):
            raise CompileError("metadata arguments do not match the kernel inputs")
        if not {t.name for t in node.outputs}.issubset(args):
            raise CompileError("metadata is missing a kernel output")
        for i, name in enumerate(self.params):
            symbol(name)
            if args[name].get("scalar"):
                raise CompileError("by-value scalar parameters are not supported; constants must be buffers")
            if args[name]["dtype"] != "f32" or self.buffers.get(name, (None, None))[1] != "f32":
                raise CompileError(f"buffer {name}: this backend supports f32 data only")
            pointer = self.reg("u64")
            self.emit(f"ld.param.u64 {pointer.text}, [param_{i}];")
            self.emit(f"cvta.to.global.u64 {pointer.text}, {pointer.text};")
            self.pointers[name] = pointer
        self.prepare_shared()
        self.tid = self.instruction("s32", "mov.u32", "%tid.x", bounds=(0, launch["block"][0]-1))
        self.env["lane"] = self.instruction("s32", "and.b32", self.tid, 31)
        self.env["warp"] = self.instruction("s32", "shr.u32", self.tid, 5)
        self.bid = self.instruction("s32", "mov.u32", "%ctaid.x")
        self.ntid = self.instruction("s32", "mov.u32", "%ntid.x")

    def emit(self, line: str):
        self.lines.append("    " + line)

    def reg(self, dtype: str, bounds=(None, None)) -> Value:
        prefix = REGISTER_PREFIX[dtype]
        self.counts[prefix] += 1
        return Value(f"%{prefix}{self.counts[prefix]}", dtype, None, *bounds)

    def label(self, prefix="L") -> str:
        self.labels += 1
        return f"{prefix}_{self.labels}"

    @staticmethod
    def operand(value) -> str:
        return value.text if isinstance(value, Value) else str(value)

    def instruction(self, dtype, opcode, *args, bounds=(None, None)) -> Value:
        result = self.reg(dtype, bounds)
        operands = ", ".join(self.operand(a) for a in args)
        tail = f", {operands}" if operands else ""
        self.emit(f"{opcode} {result.text}{tail};")
        return result

    constant = staticmethod(constant)

    def bind(self, name, value: Value):
        if name not in self.env:
            self.env[name] = self.reg(value.dtype)
        dest = self.env[name]
        if dest.dtype != value.dtype:
            raise CompileError(f"inconsistent types for {name}: {dest.dtype} and {value.dtype}")
        mov_type = MOVE_TYPES[value.dtype]
        self.emit(f"mov.{mov_type} {dest.text}, {value.text};")
        self.env[name] = Value(dest.text, dest.dtype, None, value.minimum, value.maximum)

    def expr(self, expr, cache=None) -> Value:
        # Share only within this pure expression, never across assignments or loop edges.
        if cache is None:
            cache = {}
        if isinstance(expr, Var):
            if expr.name not in self.env:
                raise CompileError(f"unbound IR value {expr.name!r}")
            return self.env[expr.name]
        if isinstance(expr, Literal):
            return self.constant(expr.value, "s32" if expr.dtype in ("int", "i32", "bool") else "f32")
        if isinstance(expr, BinaryExpr):
            left, right = self.expr(expr.left, cache), self.expr(expr.right, cache)
            key = (expr.op, left, right)
            if key not in cache:
                cache[key] = self.binary(expr.op, left, right)
            return cache[key]
        if isinstance(expr, TernaryExpr):
            pred = self.predicate(self.expr(expr.cond, cache))
            yes, no = self.expr(expr.if_true, cache), self.expr(expr.if_false, cache)
            if yes.dtype != no.dtype:
                raise CompileError("ternary branches have different types")
            if yes.dtype == "pred":
                a = self.instruction("pred", "and.pred", pred, yes)
                inverse = self.instruction("pred", "not.pred", pred)
                b = self.instruction("pred", "and.pred", inverse, no)
                return self.instruction("pred", "or.pred", a, b)
            return self.instruction(yes.dtype, f"selp.{yes.dtype}", yes, no, pred)
        raise CompileError(f"unsupported expression {type(expr).__name__}")

    def predicate(self, value: Value) -> Value:
        if value.dtype == "pred":
            return value
        return self.instruction("pred", f"setp.ne.{value.dtype}", value, self.constant(0, value.dtype))

    def binary(self, op, left: Value, right: Value) -> Value:
        if op in ("&&", "||"):
            return self.instruction("pred", "and.pred" if op == "&&" else "or.pred",
                                    self.predicate(left), self.predicate(right))
        if left.dtype != right.dtype:
            raise CompileError(f"mixed expression types: {left.dtype}, {right.dtype}")
        if left.dtype == "pred":
            if op not in ("==", "!="):
                raise CompileError(f"unsupported predicate operator {op}")
            different = self.instruction("pred", "xor.pred", left, right)
            return self.instruction("pred", "not.pred", different) if op == "==" else different
        if op in COMPARISONS:
            return self.instruction("pred", f"setp.{COMPARISONS[op]}.{left.dtype}", left, right)
        dtype = left.dtype
        if dtype == "f32":
            code = FLOAT_OPS.get(op)
            if code is None:
                raise CompileError(f"unsupported float operator {op}")
        elif dtype == "s32":
            folded = self.fold_integer(op, left, right)
            if folded is not None:
                return folded
            if op in ("/", "//", "%"):
                return self.integer_divmod(op, left, right)
            code = INTEGER_OPS.get(op)
            if code is None:
                raise CompileError(f"unsupported binary operator {op}")
        else:
            raise CompileError(f"unsupported arithmetic type {dtype}")
        suffix = "b32" if op in ("^", "&", "|", "<<") else dtype
        bounds = self.integer_bounds(op, left, right) if dtype == "s32" else (None, None)
        return self.instruction(dtype, f"{code}.{suffix}", left, right, bounds=bounds)

    @staticmethod
    def integer_bounds(op, left, right):
        if None in (left.minimum, left.maximum, right.minimum, right.maximum):
            return None, None
        if op not in ("+", "-", "*"):
            return None, None
        operation = INTEGER_FOLD[op]
        values = [operation(a, b) for a in (left.minimum, left.maximum) for b in (right.minimum, right.maximum)]
        lo, hi = min(values), max(values)
        return (lo, hi) if -(1 << 31) <= lo <= hi < (1 << 31) else (None, None)

    def fold_integer(self, op, left, right):
        a, b = left.integer, right.integer
        if op in ("/", "//", "%") and b == 0:
            raise CompileError("integer division by zero")
        if a is not None and b is not None:
            if op in INTEGER_FOLD:
                value = INTEGER_FOLD[op](a, b)
                value = (value + (1 << 31)) % (1 << 32) - (1 << 31)
                return self.constant(value, "s32")
        if op == "+" and a == 0 or op == "*" and a == 1:
            return right
        if op in ("+", "-") and b == 0 or op == "*" and b == 1 or op in ("/", "//") and b == 1:
            return left
        if op == "*" and (a == 0 or b == 0) or op == "%" and b == 1:
            return self.constant(0, "s32")
        return None

    def integer_divmod(self, op, left, right):
        # PTX truncates toward zero; Emmy integer /, // and % use floor semantics.
        if left.minimum is not None and left.minimum >= 0 and right.minimum is not None and right.minimum > 0:
            code = "rem" if op == "%" else "div"
            upper = right.maximum-1 if op == "%" else left.maximum // right.minimum
            return self.instruction("s32", f"{code}.s32", left, right, bounds=(0, upper))
        remainder = self.instruction("s32", "rem.s32", left, right)
        nonzero = self.instruction("pred", "setp.ne.s32", remainder, 0)
        signs = self.instruction("s32", "xor.b32", left, right)
        negative = self.instruction("pred", "setp.lt.s32", signs, 0)
        adjust = self.instruction("pred", "and.pred", nonzero, negative)
        if op == "%":
            correction = self.instruction("s32", "selp.s32", right, 0, adjust)
            return self.instruction("s32", "add.s32", remainder, correction)
        quotient = self.instruction("s32", "div.s32", left, right)
        correction = self.instruction("s32", "selp.s32", 1, 0, adjust)
        return self.instruction("s32", "sub.s32", quotient, correction)

    def compute(self, op: str, args: list[Value]) -> Value:
        if any(a.dtype != "f32" for a in args):
            raise CompileError(f"{op}: only f32 arithmetic is supported")
        if op in ELEMENTWISE_BINARY:
            return self.binary(ELEMENTWISE_BINARY[op], *args)
        if op == "copy":
            return args[0]
        if op in ("maximum", "minimum"):
            return self.instruction("f32", "max.f32" if op == "maximum" else "min.f32", *args)
        if op == "relu":
            return self.instruction("f32", "max.f32", args[0], self.constant(0))
        if op == "reciprocal":
            return self.binary("/", self.constant(1), args[0])
        if op == "sqrt":
            return self.instruction("f32", "sqrt.rn.f32", *args)
        if op == "rsqrt":
            return self.instruction("f32", "rsqrt.approx.f32", *args)
        if op in ("exp", "log"):
            if op == "exp":
                scaled = self.binary("*", args[0], self.constant(math.log2(math.e)))
                return self.instruction("f32", "ex2.approx.f32", scaled)
            value = self.instruction("f32", "lg2.approx.f32", args[0])
            return self.binary("*", value, self.constant(math.log(2)))
        if op in ("negative", "neg", "abs"):
            return self.instruction("f32", "abs.f32" if op == "abs" else "neg.f32", *args)
        raise CompileError(f"unsupported elementwise operation {op!r}")

    def prepare_shared(self):
        offset = 0
        specs = {}
        for stmt in walk(self.kernel.body):
            if not isinstance(stmt, Smem):
                continue
            symbol(stmt.name)
            if stmt.dtype not in ("float", "f32"):
                raise CompileError(f"shared buffer {stmt.name}: unsupported type {stmt.dtype}")
            alignment = max(stmt.align, 4)
            if alignment & (alignment - 1):
                raise CompileError("shared memory alignment must be a power of two")
            offset = (offset + alignment - 1) // alignment * alignment
            if stmt.name in specs:
                raise CompileError(f"duplicate shared buffer {stmt.name}")
            specs[stmt.name] = (stmt, offset)
            self.buffers[stmt.name] = (stmt.extents, "f32")
            offset += math.prod(stmt.extents) * 4
        dynamic = offset > 48 * 1024
        if dynamic and self.launch["smem_bytes"] < offset:
            raise CompileError(f"launch needs at least {offset} bytes of dynamic shared memory")
        if dynamic:
            alignment = max(16, *(max(stmt.align, 4) for stmt, _ in specs.values()))
            self.module_declarations.append(f".extern .shared .align {alignment} .b8 shared_pool[];")
        for name, (stmt, start) in specs.items():
            base = self.reg("s32")
            self.shared[name] = base
            if dynamic:
                self.emit(f"mov.u32 {base.text}, shared_pool;")
                self.emit(f"add.u32 {base.text}, {base.text}, {start};")
            else:
                self.declarations.append(f".shared .align {max(stmt.align, 4)} .b8 smem_{name}[{math.prod(stmt.extents) * 4}];")
                self.emit(f"mov.u32 {base.text}, smem_{name};")

    def address(self, buffer: str, index) -> Value:
        shape, _ = self.buffers[buffer]
        if len(index) != len(shape) and not (len(index) == 0 and math.prod(shape) == 1):
            raise CompileError(f"{buffer}: index rank {len(index)} differs from shape {shape}")
        flat = self.constant(0, "s32")
        for i, coordinate in enumerate(index):
            flat = self.binary("+", self.binary("*", flat, self.constant(shape[i], "s32")), self.expr(coordinate))
        if buffer in self.shared:
            offset = self.instruction("s32", "mul.lo.u32", flat, 4)
            return self.instruction("s32", "add.u32", self.shared[buffer], offset)
        offset = self.instruction("u64", "mul.wide.s32", flat, 4)
        return self.instruction("u64", "add.u64", self.pointers[buffer], offset)

    def load(self, buffer, index, names):
        address = self.address(buffer, index)
        space = "shared" if buffer in self.shared else "global"
        if len(names) not in (1, 2, 4):
            raise CompileError("loads must contain 1, 2 or 4 values")
        regs = [self.reg("f32") for _ in names]
        suffix = "" if len(names) == 1 else f".v{len(names)}"
        dest = regs[0].text if len(names) == 1 else "{" + ", ".join(r.text for r in regs) + "}"
        self.emit(f"ld.{space}{suffix}.f32 {dest}, [{address.text}];")
        for name, reg in zip(names, regs):
            self.bind(name, reg)

    def store(self, buffer, index, values, atomic=False):
        if any(self.env[name].dtype != "f32" for name in values):
            raise CompileError("float32 stores require float32 values")
        address = self.address(buffer, index)
        space = "shared" if buffer in self.shared else "global"
        if atomic:
            if space != "global":
                raise CompileError("only global atomic adds are supported")
            for i, name in enumerate(values):
                self.emit(f"red.global.add.f32 [{address.text}+{i * 4}], {self.env[name].text};")
            return
        if len(values) not in (1, 2, 4):
            raise CompileError("stores must contain 1, 2 or 4 values")
        suffix = "" if len(values) == 1 else f".v{len(values)}"
        src = self.env[values[0]].text if len(values) == 1 else "{" + ", ".join(self.env[v].text for v in values) + "}"
        self.emit(f"st.{space}{suffix}.f32 [{address.text}], {src};")

    def barrier(self, barrier_id=0, count=None, warp=False):
        if warp:
            mask = self.instruction("s32", "activemask.b32")
            self.emit(f"bar.warp.sync {mask.text};")
        else:
            tail = "" if count is None else f", {count}"
            self.emit(f"bar.sync {barrier_id}{tail};")

    def loop_bindings(self, body):
        local, accumulators = set(), {}
        for stmt in body:
            if isinstance(stmt, Accum):
                accumulators[stmt.name] = stmt
            elif isinstance(stmt, Load):
                local.update(stmt.names)
            elif isinstance(stmt, (Assign, Init, Let)):
                local.add(stmt.name)
            elif isinstance(stmt, Cond):
                for branch in (stmt.body, stmt.else_body):
                    names, states = self.loop_bindings(branch)
                    local.update(names)
                    accumulators.update(states)
        return local - accumulators.keys(), accumulators

    def loop(self, stmt):
        strided = isinstance(stmt, StridedLoop)
        start = self.expr(stmt.start) if strided else self.constant(0, "s32")
        step = self.expr(stmt.step) if strided else self.constant(1, "s32")
        end = self.expr(stmt.end) if strided and stmt.end is not None else self.constant(stmt.axis.extent.as_static(), "s32")
        if start.dtype != "s32" or step.dtype != "s32" or end.dtype != "s32":
            raise CompileError("loop bounds must be integers")
        if step.integer is not None and step.integer <= 0:
            raise CompileError("loop stride must be positive")
        outer = self.env
        self.env = outer.copy()
        local, accumulators = self.loop_bindings(stmt.body)
        for name in local | {stmt.axis.name}:
            self.env.pop(name, None)
        if stmt.seed:
            for name, accumulator in accumulators.items():
                if accumulator.op.identity is None:
                    raise CompileError(f"{accumulator.op.name} has no reduction identity")
                self.env.pop(name, None)
                self.bind(name, self.constant(accumulator.op.identity))
        self.bind(stmt.axis.name, start)
        induction = self.env[stmt.axis.name]
        if start.minimum is not None and end.maximum is not None and start.minimum < end.maximum:
            induction = Value(induction.text, induction.dtype, None, start.minimum, end.maximum-1)
            self.env[stmt.axis.name] = induction
        top, done = self.label("loop"), self.label("done")
        self.emit(top + ":")
        stop = self.binary(">=", induction, end)
        self.emit(f"@{stop.text} bra {done};")
        self.body(stmt.body)
        self.emit(f"add.s32 {induction.text}, {induction.text}, {step.text};")
        self.emit(f"bra {top};")
        self.emit(done + ":")
        exports = {name: self.env[name] for name in accumulators}
        self.env = outer
        self.env.update(exports)

    def conditional(self, pred: Value, body, else_body=()):
        other, done = self.label("else"), self.label("endif")
        self.emit(f"@!{pred.text} bra {other};")
        self.body(body)
        self.emit(f"bra {done};")
        self.emit(other + ":")
        self.body(else_body)
        self.emit(done + ":")

    def tile(self, stmt: Tile):
        # Raster hints only permute independent CTAs. Row-major visits the same cells.
        block = stmt.block_threads
        if stmt.aux_threads and not block:
            raise CompileError("auxiliary threads require block_threads")
        local = self.instruction("s32", "rem.u32", self.tid, block) if stmt.aux_threads else self.tid
        gid = self.instruction("s32", "mad.lo.s32", self.bid, block or self.ntid, local)
        extents = [a.extent.as_static() for a in stmt.axes]
        total = math.prod(extents)
        if any(e <= 0 for e in extents) or total >= (1 << 31):
            raise CompileError("tile extents must be positive and fit signed 32-bit indexing")
        done = self.label("tile_end")
        out = self.binary(">=", gid, self.constant(total, "s32"))
        self.emit(f"@{out.text} bra {done};")
        stride = total
        for axis, extent in zip(stmt.axes, extents):
            stride //= extent
            coordinate = self.instruction("s32", "div.u32", gid, stride)
            coordinate = self.instruction("s32", "rem.u32", coordinate, extent, bounds=(0, extent-1))
            self.bind(axis.name, coordinate)
        self.body(stmt.body)
        self.emit(done + ":")

    def shuffle(self, stmt: WarpShuffle):
        if not stmt.state or len(stmt.state) != len(stmt.state_b):
            raise CompileError("shuffle state component counts must match")
        if stmt.dtype.name != "f32" or stmt.length < 1 or stmt.length > 32 or stmt.length & (stmt.length - 1):
            raise CompileError("warp shuffle needs f32 and a power of two length up to 32")
        mask = self.instruction("s32", "activemask.b32")
        distance = stmt.length // 2
        while distance:
            for name, other in zip(stmt.state, stmt.state_b):
                value = self.instruction("f32", "shfl.sync.bfly.b32", self.env[name], distance, 31, mask)
                self.bind(other, value)
            self.body(stmt.combine_states)
            distance //= 2

    def tree(self, stmt: TreeHalve):
        if not stmt.state or not len(stmt.bufs) == len(stmt.state) == len(stmt.state_b):
            raise CompileError("shared tree state component counts must match")
        if stmt.length < 1 or stmt.length & (stmt.length - 1):
            raise CompileError("shared reduction length must be a power of two")
        tid = self.env[stmt.tid_var]
        scale = stmt.inner[1] if stmt.inner else 1
        inner = Var(stmt.inner[0]) if stmt.inner else Literal(0, "int")
        def index(offset):
            return (BinaryExpr("+", BinaryExpr("*", offset, Literal(scale, "int")), inner),)
        distance = stmt.length // 2
        while distance:
            skip = self.label("tree_skip")
            active = self.binary("<", tid, self.constant(distance, "s32"))
            if stmt.tid_var == "warp":
                leader = self.binary("==", self.env["lane"], self.constant(0, "s32"))
                active = self.binary("&&", active, leader)
            self.emit(f"@!{active.text} bra {skip};")
            for buf, name, other in zip(stmt.bufs, stmt.state, stmt.state_b):
                self.load(buf, index(Var(stmt.tid_var)), (name,))
                self.load(buf, index(BinaryExpr("+", Var(stmt.tid_var), Literal(distance, "int"))), (other,))
            self.body(stmt.combine_states)
            for buf, name in zip(stmt.bufs, stmt.state):
                self.store(buf, index(Var(stmt.tid_var)), (name,))
            self.emit(skip + ":")
            self.barrier(stmt.barrier_id, stmt.barrier_count)
            distance //= 2
        for buf, name in zip(stmt.bufs, stmt.state):
            self.load(buf, (inner,), (name,))
        self.barrier(stmt.barrier_id, stmt.barrier_count)

    def typed_value(self, value, dtype):
        target = getattr(dtype, "name", dtype)
        target = {"i32": "s32", "int": "s32"}.get(target, target)
        if target is None or target == value.dtype:
            return value
        if target == "f32" and value.dtype == "s32":
            if value.integer is not None:
                return self.constant(value.integer)
            return self.instruction("f32", "cvt.rn.f32.s32", value)
        if target == "s32" and value.dtype == "f32":
            return self.instruction("s32", "cvt.rzi.s32.f32", value)
        raise CompileError(f"unsupported value conversion {value.dtype} to {target}")

    def body(self, body):
        for stmt in body:
            self.emit(f"// {type(stmt).__name__}")
            dtype = getattr(stmt, "dtype", getattr(stmt, "value_dtype", None))
            if not isinstance(stmt, (Let, Smem)) and dtype is not None and dtype.name != "f32":
                raise CompileError(f"{type(stmt).__name__}: only f32 data is supported")
            if isinstance(stmt, Tile):
                self.tile(stmt)
            elif isinstance(stmt, (Loop, StridedLoop)):
                self.loop(stmt)
            elif isinstance(stmt, Load):
                self.load(stmt.input, stmt.index, stmt.names)
            elif isinstance(stmt, Write):
                if stmt.swizzle != "NONE":
                    raise CompileError("swizzled stores are not supported")
                self.store(stmt.output, stmt.index, stmt.values, stmt.atomic)
            elif isinstance(stmt, Assign):
                self.bind(stmt.name, self.compute(stmt.op.name, [self.env[a] for a in stmt.args]))
            elif isinstance(stmt, Accum):
                base = stmt.base or stmt.name
                self.bind(stmt.name, self.compute(stmt.op.name, [self.env[base], self.env[stmt.value]]))
            elif isinstance(stmt, Init):
                self.bind(stmt.name, self.constant(stmt.identity))
            elif isinstance(stmt, Let):
                self.bind(stmt.name, self.typed_value(self.expr(stmt.value), stmt.dtype))
            elif isinstance(stmt, Cond):
                self.conditional(self.predicate(self.expr(stmt.cond)), stmt.body, stmt.else_body)
            elif isinstance(stmt, Smem):
                pass  # All shared declarations and base pointers are prepared in the prologue.
            elif isinstance(stmt, Sync):
                self.barrier(stmt.barrier_id, stmt.count, stmt.warp)
            elif isinstance(stmt, WarpShuffle):
                self.shuffle(stmt)
            elif isinstance(stmt, TreeHalve):
                self.tree(stmt)
            elif isinstance(stmt, CpAsyncCopy):
                if stmt.swizzle != "NONE":
                    raise CompileError("swizzled async copies are not supported")
                if stmt.nbytes not in (4, 8, 16):
                    raise CompileError("async copies must contain 4, 8 or 16 bytes")
                index = stmt.smem_index
                if stmt.lane_index is not None:
                    if len(index) != len(stmt.lane_index):
                        raise CompileError("async copy lane index rank differs from slab rank")
                    index = tuple(BinaryExpr("+", a, b) for a, b in zip(index, stmt.lane_index))
                dst = self.address(stmt.smem, index)
                src = self.address(stmt.src, stmt.src_index)
                cache = "cg" if stmt.nbytes == 16 else "ca"
                self.emit(f"cp.async.{cache}.shared.global [{dst.text}], [{src.text}], {stmt.nbytes};")
            elif isinstance(stmt, CpAsyncCommit):
                self.emit("cp.async.commit_group;")
            elif isinstance(stmt, CpAsyncWait):
                self.emit(f"cp.async.wait_group {stmt.group};")
            else:
                raise CompileError(f"unsupported statement {type(stmt).__name__}")

    def compile(self) -> str:
        self.body(self.kernel.body)
        self.emit("ret;")
        entry = symbol(self.kernel.name)
        params = ",\n".join(f"    .param .u64 param_{i}" for i in range(len(self.params)))
        header = [".version 8.0", ".target sm_89", ".address_size 64", "", *self.module_declarations,
                  f".visible .entry {entry}(\n{params}\n)", "{"]
        types = {"r": "b32", "rd": "b64", "f": "f32", "p": "pred"}
        declarations = [f"    .reg .{types[p]} %{p}<{n + 1}>;" for p, n in self.counts.items()]
        source = "\n".join(header + declarations + ["    " + d for d in self.declarations] + self.lines + ["}", ""])
        source.encode("ascii")
        return source


def compile_file(path: Path | str) -> str:
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    nodes = [node for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
    if len(nodes) != 1:
        raise CompileError(f"expected one KernelOp, found {len(nodes)}")
    metadata = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
    matches = [k for k in metadata["kernels"] if k["entry"] == nodes[0].op.name]
    if len(matches) != 1:
        raise CompileError("meta.json must contain exactly one matching entry")
    return Compiler(graph, nodes[0], matches[0]).compile()
