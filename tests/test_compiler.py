"""Semantic regressions on fresh typed IRs, with optional real GPU execution.

Run CPU checks with python -m unittest discover -s tests -v.
Set EMMY_PTX_GPU_TESTS=1 to assemble and execute the synthetic cases too.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write
from emmy.compiler.tensor import Tensor

from emmy_ptx import CompileError, compile_file, constant
from tools.validate import Cuda, validate

ROOT = Path(__file__).resolve().parents[1]


def axis(name, extent):
    return Axis(name, Dim(extent))


def binary(op, left, right):
    return BinaryExpr(op, left, Literal(right, "int"))


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, inputs, output_shape, body, expected, block, grid=1):
        graph = Graph()
        graph.inputs, graph.outputs = list(inputs), ["answer"]
        descriptors = []
        for name, data in inputs.items():
            data = np.asarray(data, dtype=np.float32)
            tensor = Tensor(name, tuple(Dim(d) for d in data.shape), F32)
            graph.nodes[name] = Node(name, InputOp(), [], (tensor,))
            descriptors.append({"name": name, "shape": list(data.shape), "dtype": "f32", "data": data.ravel().tolist()})
        output = Tensor("answer", tuple(Dim(d) for d in output_shape), F32)
        graph.nodes["answer"] = Node("answer", KernelOp(body=(body,), name="fresh_kernel"), list(inputs), (output,))
        descriptors.append({"name": "answer", "shape": list(output_shape), "dtype": "f32", "output": True})
        # Outputs first and reversed inputs exercise the positional ABI independently of graph order.
        descriptors.reverse()
        kernel = {"entry": "fresh_kernel", "arg_order": [a["name"] for a in descriptors], "args": descriptors,
                  "block": [block, 1, 1], "grid": [grid, 1, 1], "smem_bytes": 0, "zero_outputs": []}
        metadata = {"kernels": [kernel], "graph_outputs": ["answer"], "rtol": 1e-5, "atol": 1e-5}
        path = self.folder / "fresh_kernel.ir.json"
        path.write_text(json.dumps(graph.to_dict(), default=str), encoding="utf-8")
        (self.folder / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez(self.folder / "expected.npz", answer=np.asarray(expected, dtype=np.float32))
        return path

    def check_fixture(self, path):
        source = compile_file(path)
        self.assertTrue(source.isascii())
        self.assertTrue(source.startswith(".version 8.0\n"))
        if os.environ.get("EMMY_PTX_GPU_TESTS") == "1":
            cuda = Cuda()
            try:
                result = validate(self.folder, self.folder, "ptxas", "sm_89", cuda)
                self.assertEqual(result["status"], "PASS")
            finally:
                cuda.close()

    def test_vector_loads_nondivisible_tile_and_reordered_arguments(self):
        left = np.arange(140, dtype=np.float32).reshape(7, 20)
        right = np.full((7, 20), 17, dtype=np.float32)
        index = (Var("row"), binary("*", Var("col"), 4))
        body = [Load(input="left", index=index, names=tuple(f"l{i}" for i in range(4)), dtype=F32),
                Load(input="right", index=index, names=tuple(f"r{i}" for i in range(4)), dtype=F32)]
        body += [Assign(f"v{i}", "subtract", (f"l{i}", f"r{i}"), F32) for i in range(4)]
        body += [Write("answer", index, values=tuple(f"v{i}" for i in range(4)), value_dtype=F32)]
        tile = Tile((axis("row", 7), axis("col", 5)), tuple(body))
        path = self.fixture({"left": left, "right": right}, (7, 20), tile, left-right, block=32, grid=2)
        self.check_fixture(path)

    def test_maximum_reduction_with_empty_thread_partials(self):
        data = -np.arange(1, 186, dtype=np.float32).reshape(5, 37)
        combine = (Assign("acc", "maximum", ("acc", "other"), F32),)
        loop = StridedLoop(axis("j", 37), Var("co"), Literal(64, "int"), (
            Load("x", "input", (Var("row"), Var("j")), dtype=F32),
            Accum("acc", "x", "maximum", F32, ("j",)),))
        tile = Tile((axis("row", 5), axis("co", 64)), (
            loop, WarpShuffle(("acc",), ("other",), combine, 32), Smem("partials", (2,)),
            Cond(binary("==", Var("lane"), 0), (Write("partials", (Var("warp"),), "acc", value_dtype=F32),)),
            Sync(), TreeHalve(("partials",), ("acc",), ("other",), combine, 2, "warp"),
            Cond(binary("==", Var("co"), 0), (Write("answer", (Var("row"),), "acc", value_dtype=F32),)),
        ), block_threads=64)
        path = self.fixture({"input": data}, (5,), tile, data.max(axis=1), block=64, grid=5)
        self.check_fixture(path)

    def test_seed_false_keeps_accumulator_across_nested_loops(self):
        data = np.arange(11, dtype=np.float32)
        inner = Loop(axis("inner", 3), (Accum("acc", "x", "add", F32),), seed=False)
        outer = Loop(axis("outer", 2), (inner,))
        tile = Tile((axis("i", 11),), (Load("x", "input", (Var("i"),), dtype=F32),
                    Init("acc", 0, dtype=F32), outer, Write("answer", (Var("i"),), "acc", value_dtype=F32)))
        path = self.fixture({"input": data}, (11,), tile, data*6, block=32)
        self.check_fixture(path)

    def test_cli_is_ptx_only_and_reports_errors_on_stderr(self):
        data = np.arange(3, dtype=np.float32)
        tile = Tile((axis("i", 3),), (Load("x", "input", (Var("i"),), dtype=F32),
                    Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({"input": data}, (3,), tile, data, block=32)
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, compile_file(path))
        (self.folder / "meta.json").unlink()
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("meta.json", result.stderr)

    def test_rejects_unsupported_math_without_partial_ptx(self):
        data = np.ones(1, dtype=np.float32)
        tile = Tile((axis("i", 1),), (Load("x", "input", (Var("i"),), dtype=F32),
                    Assign("y", "sin", ("x",), F32), Write("answer", (Var("i"),), "y", value_dtype=F32)))
        path = self.fixture({"input": data}, (1,), tile, np.sin(data), block=32)
        with self.assertRaisesRegex(CompileError, "unsupported elementwise"):
            compile_file(path)

    def test_nested_loop_local_does_not_overwrite_outer_value(self):
        tile = Tile((axis("i", 3),), (
            Let("saved", Literal(2.0), F32),
            Loop(axis("j", 2), (Let("saved", Literal(9.0), F32),)),
            Write("answer", (Var("i"),), "saved", value_dtype=F32)))
        path = self.fixture({}, (3,), tile, np.full(3, 2.0), block=32)
        self.check_fixture(path)

    def test_signed_index_division_and_modulo(self):
        data = np.arange(12, dtype=np.float32)
        negative = binary("-", Var("i"), 5)
        for divisor in (3, -3):
            for op in ("/", "//", "%"):
                index = binary("+", binary(op, negative, divisor), 3)
                expected_index = ((np.arange(8)-5) % divisor if op == "%" else (np.arange(8)-5)//divisor)+3
                tile = Tile((axis("i", 8),), (Load("x", "input", (index,), dtype=F32),
                            Write("answer", (Var("i"),), "x", value_dtype=F32)))
                path = self.fixture({"input": data}, (8,), tile, data[expected_index], block=32)
                self.check_fixture(path)

    def test_predicate_comparison_and_predicate_ternary(self):
        less = binary("<", Var("i"), 2)
        greater = binary(">", Var("i"), 0)
        equal = BinaryExpr("==", less, greater)
        choice = TernaryExpr(less, equal, greater)
        value = TernaryExpr(choice, Literal(1.0), Literal(-1.0))
        tile = Tile((axis("i", 4),), (Let("x", value, F32),
                    Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({}, (4,), tile, [-1, 1, 1, 1], block=32)
        self.check_fixture(path)

    def test_let_honors_declared_f32_dtype_for_integer_literal(self):
        tile = Tile((axis("i", 3),), (Let("x", Literal(2, "int"), F32),
                    Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({}, (3,), tile, np.full(3, 2.0), block=32)
        self.check_fixture(path)

    def test_rejects_nonpositive_static_stride(self):
        tile = Tile((axis("i", 1),), (StridedLoop(axis("j", 2), Literal(0, "int"), Literal(0, "int"), ()),
                    Let("x", Literal(1.0), F32), Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({}, (1,), tile, [1], block=32)
        with self.assertRaisesRegex(CompileError, "positive"):
            compile_file(path)

    def test_rejects_mismatched_reduction_components(self):
        tile = Tile((axis("i", 32),), (Init("acc", 0, dtype=F32),
                    WarpShuffle(("acc",), (), (), 32),))
        path = self.fixture({}, (1,), tile, [0], block=32)
        with self.assertRaisesRegex(CompileError, "component"):
            compile_file(path)

    def test_worker_recovers_after_an_error_and_matches_single_file_compilation(self):
        tile = Tile((axis("i", 2),), (Let("x", Literal(7.0), F32),
                    Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({}, (2,), tile, [7, 7], block=32)
        requests = ["not json", json.dumps({"ir": str(path)}), json.dumps({"ir": str(path)})]
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), "--worker"],
                                input="\n".join(requests)+"\n", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(responses), 3)
        self.assertFalse(responses[0]["ok"])
        for response in responses[1:]:
            self.assertTrue(response["ok"])
            self.assertEqual(response["ptx"], compile_file(path))

    def test_literal_cache_preserves_signed_zero(self):
        import struct
        positive = constant(0.0).text[2:]
        negative = constant(-0.0).text[2:]
        self.assertEqual(struct.unpack("!f", bytes.fromhex(positive))[0], 0.0)
        self.assertTrue(np.signbit(struct.unpack("!f", bytes.fromhex(negative))[0]))

    def test_rejects_integer_register_in_float_store(self):
        from emmy.compiler.dtype import I32
        tile = Tile((axis("i", 1),), (Let("x", Literal(7, "int"), I32),
                    Write("answer", (Var("i"),), "x", value_dtype=F32)))
        path = self.fixture({}, (1,), tile, [7], block=32)
        with self.assertRaisesRegex(CompileError, "float32 stores"):
            compile_file(path)


if __name__ == "__main__":
    unittest.main()
