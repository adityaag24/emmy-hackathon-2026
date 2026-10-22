#!/usr/bin/env python3
"""Assemble and optionally execute every open problem without CUDA C++ or torch.

    python tools/validate.py
    python tools/validate.py --assemble-only
    python tools/validate.py p00_scalar_add r01_array_sum
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy_ptx import compile_file


class Cuda:
    """Small CUDA Driver API host, used only to validate the emitted cubins."""

    def __init__(self):
        self.lib = ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        pointer, integer, size, deviceptr = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_uint64
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDeviceGet": [ct.POINTER(integer), integer],
            "cuDeviceGetName": [ct.c_char_p, integer, integer],
            "cuCtxCreate_v2": [ct.POINTER(pointer), ct.c_uint, integer],
            "cuCtxDestroy_v2": [pointer],
            "cuModuleLoadData": [ct.POINTER(pointer), pointer],
            "cuModuleUnload": [pointer],
            "cuModuleGetFunction": [ct.POINTER(pointer), pointer, ct.c_char_p],
            "cuFuncSetAttribute": [pointer, integer, integer],
            "cuMemAlloc_v2": [ct.POINTER(deviceptr), size],
            "cuMemFree_v2": [deviceptr],
            "cuMemcpyHtoD_v2": [deviceptr, pointer, size],
            "cuMemcpyDtoH_v2": [pointer, deviceptr, size],
            "cuLaunchKernel": [pointer] + [ct.c_uint] * 7 + [pointer, ct.POINTER(pointer), pointer],
            "cuCtxSynchronize": [],
            "cuGetErrorName": [integer, ct.POINTER(ct.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, integer
        self.call("cuInit", 0)
        device = integer()
        self.call("cuDeviceGet", ct.byref(device), 0)
        name = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        self.name = name.value.decode()
        self.context = pointer()
        self.call("cuCtxCreate_v2", ct.byref(self.context), 0, device)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            message = ct.c_char_p()
            self.lib.cuGetErrorName(status, ct.byref(message))
            raise RuntimeError(f"{name}: {(message.value or str(status).encode()).decode()}")

    def close(self):
        self.call("cuCtxDestroy_v2", self.context)

    def run(self, cubin: Path, kernel: dict, inputs: dict) -> dict:
        module, function = ct.c_void_p(), ct.c_void_p()
        image = ct.create_string_buffer(cubin.read_bytes())
        self.call("cuModuleLoadData", ct.byref(module), image)
        allocations, arrays = [], {}
        try:
            self.call("cuModuleGetFunction", ct.byref(function), module, kernel["entry"].encode("ascii"))
            if kernel["smem_bytes"] > 48 * 1024:
                self.call("cuFuncSetAttribute", function, 8, kernel["smem_bytes"])
            for arg in kernel["args"]:
                name, shape = arg["name"], arg["shape"]
                if arg.get("output") or name in kernel["zero_outputs"]:
                    array = np.zeros(shape, dtype=np.float32)
                elif arg.get("intermediate"):
                    array = np.asarray(inputs[name], dtype=np.float32).reshape(shape)
                elif "data" in arg:
                    array = np.asarray(arg["data"], dtype=np.float32).reshape(shape)
                else:
                    array = np.random.default_rng(arg["seed"]).standard_normal(shape).astype(np.float32)
                array = np.ascontiguousarray(array)
                pointer = ct.c_uint64()
                self.call("cuMemAlloc_v2", ct.byref(pointer), array.nbytes)
                allocations.append(pointer)
                self.call("cuMemcpyHtoD_v2", pointer, array.ctypes.data, array.nbytes)
                arrays[name] = (array, pointer)
            parameters = (ct.c_void_p * len(kernel["arg_order"]))(
                *(ct.cast(ct.byref(arrays[name][1]), ct.c_void_p) for name in kernel["arg_order"]))
            self.call("cuLaunchKernel", function, *kernel["grid"], *kernel["block"],
                      kernel["smem_bytes"], None, parameters, None)
            self.call("cuCtxSynchronize")
            outputs = {}
            for arg in kernel["args"]:
                if arg.get("output"):
                    array, pointer = arrays[arg["name"]]
                    self.call("cuMemcpyDtoH_v2", array.ctypes.data, pointer, array.nbytes)
                    outputs[arg["name"]] = array
            return outputs
        finally:
            for pointer in allocations:
                self.call("cuMemFree_v2", pointer)
            self.call("cuModuleUnload", module)


def validate(problem: Path, output: Path, ptxas: str, arch: str, cuda: Cuda | None) -> dict:
    metadata = json.loads((problem / "meta.json").read_text())
    produced, records = {}, []
    for kernel in metadata["kernels"]:
        ir = problem / (kernel["entry"] + ".ir.json")
        start = time.perf_counter()
        source = compile_file(ir)
        lower_ms = (time.perf_counter() - start) * 1000
        ptx = output / (problem.name + "_" + kernel["entry"] + ".ptx")
        ptx.write_text(source, encoding="ascii")
        cubin = ptx.with_suffix(".cubin")
        start = time.perf_counter()
        result = subprocess.run([ptxas, f"-arch={arch}", "-v", str(ptx), "-o", str(cubin)],
                                capture_output=True, text=True, timeout=30)
        assembly_ms = (time.perf_counter() - start) * 1000
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout or f"ptxas exited with {result.returncode}")
        record = {"entry": kernel["entry"], "lower_ms": lower_ms, "assembly_ms": assembly_ms,
                  "ptx_bytes": len(source), "ptxas": (result.stdout + result.stderr).strip(), "outputs": {}}
        if cuda is not None:
            produced.update(cuda.run(cubin, kernel, produced))
            with np.load(problem / "expected.npz") as expected:
                for name in expected.files:
                    if name not in produced:
                        continue
                    reference = expected[name]
                    actual = produced[name].reshape(reference.shape)
                    error = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
                    passed = actual.shape == reference.shape and np.allclose(
                        actual, reference, rtol=metadata["rtol"], atol=metadata["atol"])
                    record["outputs"][name] = {"pass": bool(passed), "max_abs_error": error}
                    if not passed:
                        raise AssertionError(f"{problem.name}/{name}: max absolute error {error}")
        records.append(record)
    if cuda is not None:
        missing = set(metadata["graph_outputs"]) - produced.keys()
        if missing:
            raise AssertionError(f"missing graph outputs: {sorted(missing)}")
    return {"problem": problem.name, "status": "PASS" if cuda else "ASSEMBLED", "kernels": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="*")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "validation")
    args = parser.parse_args()
    problems = [ROOT / "problems" / name for name in args.problems] if args.problems else sorted(
        path.parent for path in (ROOT / "problems").glob("*/meta.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    cuda = None if args.assemble_only else Cuda()
    report = {"gpu": cuda.name if cuda else None, "arch": args.arch, "results": []}
    try:
        for problem in problems:
            try:
                result = validate(problem, args.output, args.ptxas, args.arch, cuda)
            except Exception as exc:
                result = {"problem": problem.name, "status": "FAIL", "error": str(exc)}
            report["results"].append(result)
            print(f"{problem.name}: {result['status']}" + (f" {result['error']}" if "error" in result else ""), flush=True)
    finally:
        if cuda is not None:
            cuda.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(any(r["status"] == "FAIL" for r in report["results"]))


if __name__ == "__main__":
    sys.exit(main())
