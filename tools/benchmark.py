#!/usr/bin/env python3
"""Compare the direct backend with public nvcc on the supplied CUDA references.

python tools/benchmark.py --repetitions 5 --warmups 1 --verify
Optional --baseline-compiler points to a previous compile.py with its backend beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy_ptx import compile_file


def run(command, env, timeout=120):
    result = subprocess.run([str(part) for part in command], env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited with {result.returncode}: {result.stderr or result.stdout}")
    return result


def host_environment(output, vcvars=None):
    """Prepare MSVC once outside timing; avoid charging nvcc per-call shell setup."""
    env = os.environ.copy()
    if os.name != "nt":
        return env, None
    if vcvars is None and shutil.which("cl") and env.get("VSCMD_VER"):
        return env, "existing developer environment"
    if vcvars is None:
        locator = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        result = run([locator, "-latest", "-products", "*", "-requires",
                      "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], env)
        installation = result.stdout.strip()
        if not installation:
            raise RuntimeError("MSVC not found; pass --vcvars or run in an x64 developer shell")
        vcvars = Path(installation) / "Common7/Tools/VsDevCmd.bat"
    vcvars = Path(vcvars).resolve()
    if any(c in str(vcvars) for c in '\r\n"%'):
        raise ValueError("unsupported characters in developer environment path")
    batch = output / "developer-environment.cmd"
    batch.write_text(f'@echo off\ncall "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\n'
                     'if errorlevel 1 exit /b 1\nset\n', encoding="utf-8")
    result = run(["cmd.exe", "/d", "/c", batch.resolve()], env)
    for line in result.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            env[name] = value
    return env, str(vcvars)


class Worker:
    def __init__(self, compiler, env):
        self.process = subprocess.Popen([sys.executable, str(compiler), "--worker"], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)

    def compile(self, ir):
        self.process.stdin.write(json.dumps({"ir": str(ir)}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"compiler worker exited: {self.process.stderr.read()}")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["ptx"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()


def load_baseline(compiler):
    path = compiler.with_name("emmy_ptx.py")
    spec = importlib.util.spec_from_file_location("benchmark_baseline_emmy_ptx", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compile_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(samples):
    routes = sorted({s["route"] for s in samples})
    rows = []
    for problem, entry in sorted({(s["problem"], s["entry"]) for s in samples}):
        row = {"problem": problem, "entry": entry, "routes": {}}
        for route in routes:
            values = [s for s in samples if s["problem"] == problem and s["entry"] == entry and s["route"] == route and not s["warmup"]]
            if not values:
                continue
            times = [s["total_ms"] for s in values]
            quartiles = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
            row["routes"][route] = {
                "samples": len(times), "median_ms": statistics.median(times),
                "min_ms": min(times), "max_ms": max(times), "iqr_ms": quartiles[2]-quartiles[0],
                "lower_median_ms": statistics.median(s["lower_ms"] for s in values),
                "assembly_median_ms": statistics.median(s["assembly_ms"] for s in values),
                "ptx_bytes": values[-1]["ptx_bytes"],
            }
        rows.append(row)
    aggregate = {}
    for route in routes:
        if not all(route in row["routes"] for row in rows):
            continue
        total = sum(row["routes"][route]["median_ms"] for row in rows)
        speeds = [row["routes"]["nvcc"]["median_ms"] / row["routes"][route]["median_ms"] for row in rows]
        aggregate[route] = {"sum_of_case_medians_ms": total,
                            "geomean_speedup_vs_nvcc": math.exp(statistics.mean(math.log(s) for s in speeds))}
    nvcc_total = aggregate["nvcc"]["sum_of_case_medians_ms"]
    for data in aggregate.values():
        data["suite_speedup_vs_nvcc"] = nvcc_total / data["sum_of_case_medians_ms"]
    return rows, aggregate


def verify(cases, artifacts, routes):
    import numpy as np
    from tools.validate import Cuda
    cuda = Cuda()
    results = []
    try:
        for route in routes:
            produced = {}
            for case in cases:
                metadata = case["metadata"]
                values = produced.setdefault(case["problem"], {})
                values.update(cuda.run(artifacts[(case["problem"], case["entry"], route)], case["kernel"], values))
                with np.load(case["directory"] / "expected.npz") as expected:
                    for name in expected.files:
                        if name not in values:
                            continue
                        got, reference = values[name].reshape(expected[name].shape), expected[name]
                        passed = bool(np.allclose(got, reference, rtol=metadata["rtol"], atol=metadata["atol"]))
                        results.append({"problem": case["problem"], "route": route, "output": name,
                                        "pass": passed, "max_abs_error": float(np.max(np.abs(got-reference)))})
        return {"gpu": cuda.name, "results": results, "pass": all(row["pass"] for row in results)}
    finally:
        cuda.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="*")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--baseline-compiler", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "benchmark")
    parser.add_argument("--vcvars", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env, developer_setup = host_environment(args.output, args.vcvars)
    nvcc_flags = ["--cubin", "-O3", f"-arch={args.arch}"] + (["--use-local-env"] if os.name == "nt" else [])
    baseline = args.baseline_compiler.resolve() if args.baseline_compiler else None
    baseline_compile = load_baseline(baseline) if baseline else None
    directories = [ROOT / "problems" / p for p in args.problems] if args.problems else sorted(
        p.parent for p in (ROOT / "problems").glob("*/meta.json"))
    cases = []
    for directory in directories:
        metadata = json.loads((directory / "meta.json").read_text())
        for kernel in metadata["kernels"]:
            entry = kernel["entry"]
            cases.append({"problem": directory.name, "entry": entry, "directory": directory,
                          "ir": directory / f"{entry}.ir.json", "reference": directory / f"{entry}.ref.cu",
                          "metadata": metadata, "kernel": kernel})
    if not cases:
        raise ValueError("no benchmark cases selected")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cpu": platform.processor(), "python": sys.version,
              "emmy_version": importlib.metadata.version("emmy-ml"),
              "numpy_version": importlib.metadata.version("numpy"),
              "nvcc": run([args.nvcc, "--version"], env).stdout.strip(),
              "ptxas": run([args.ptxas, "--version"], env).stdout.strip(),
              "developer_setup": developer_setup, "nvcc_flags": nvcc_flags, "arch": args.arch,
              "repetitions": args.repetitions, "warmups": args.warmups, "seed": 48701,
              "compiler_sha256": digest(ROOT / "compile.py"), "backend_sha256": digest(ROOT / "emmy_ptx.py"),
              "benchmark_sha256": digest(Path(__file__)),
              "baseline_sha256": digest(baseline.with_name("emmy_ptx.py")) if baseline else None,
              "cases": [{"problem": c["problem"], "entry": c["entry"], "ir_sha256": digest(c["ir"]),
                         "reference_sha256": digest(c["reference"])} for c in cases], "samples": []}
    compiler = ROOT / "compile.py"
    started = time.perf_counter()
    worker = Worker(compiler, env)
    worker.compile(cases[0]["ir"])
    report["worker_startup_and_first_request_ms"] = (time.perf_counter()-started)*1000
    routes = ["api", "worker", "cli", "nvcc"] + (["baseline_api", "baseline_cli"] if baseline else [])
    artifacts = {}
    rng = random.Random(48701)
    raw = (args.output / "samples.jsonl").open("w", encoding="utf-8")
    try:
        for round_number in range(args.warmups + args.repetitions):
            jobs = [(case, route) for case in cases for route in routes]
            rng.shuffle(jobs)
            for case, route in jobs:
                tag = case["problem"] + "_" + case["entry"] + "_" + route
                ptx, cubin = args.output / (tag + ".ptx"), args.output / (tag + ".cubin")
                start = time.perf_counter()
                lower_ms, assembly_ms, ptx_bytes = 0.0, 0.0, None
                if route == "nvcc":
                    run([args.nvcc, *nvcc_flags, case["reference"], "-o", cubin], env)
                else:
                    if route == "api":
                        source = compile_file(case["ir"])
                    elif route == "baseline_api":
                        source = baseline_compile(case["ir"])
                    elif route == "worker":
                        source = worker.compile(case["ir"])
                    else:
                        source = run([sys.executable, baseline if route == "baseline_cli" else compiler, case["ir"]], env).stdout
                    lower_ms = (time.perf_counter()-start)*1000
                    ptx.write_text(source, encoding="ascii")
                    ptx_bytes = len(source)
                    before_assembly = time.perf_counter()
                    run([args.ptxas, "-O3", f"-arch={args.arch}", ptx, "-o", cubin], env)
                    assembly_ms = (time.perf_counter()-before_assembly)*1000
                sample = {"problem": case["problem"], "entry": case["entry"], "route": route,
                          "round": round_number, "warmup": round_number < args.warmups,
                          "total_ms": (time.perf_counter()-start)*1000, "lower_ms": lower_ms,
                          "assembly_ms": assembly_ms, "ptx_bytes": ptx_bytes}
                report["samples"].append(sample)
                artifacts[(case["problem"], case["entry"], route)] = cubin
                raw.write(json.dumps(sample) + "\n")
                raw.flush()
            print(f"round {round_number+1}/{args.warmups+args.repetitions} complete", flush=True)
    finally:
        raw.close()
        worker.close()
    report["summary"], report["aggregate"] = summary(report["samples"])
    report["verification"] = verify(cases, artifacts, routes) if args.verify else None
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for route, data in report["aggregate"].items():
        print(f"{route}: {data['sum_of_case_medians_ms']:.1f} ms summed case medians; "
              f"{data['suite_speedup_vs_nvcc']:.2f}x vs nvcc", flush=True)
    if report["verification"]:
        print(f"GPU checks: {len(report['verification']['results'])}; pass={report['verification']['pass']}")
        return int(not report["verification"]["pass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
