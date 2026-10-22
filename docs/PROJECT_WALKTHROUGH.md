# Emmy direct PTX compiler project and code walkthrough

The repository now contains a working compiler that translates Emmy Kernel IR
directly into PTX. `ptxas` assembles that PTX into a native GPU cubin. All 15 open
problems pass through the official RTX 5090 runner and on a local Ada GPU, using
the supplied launch geometry and tolerances. The implementation covers all five
open categories and uses no CUDA C++ generation, nvcc invocation, or external
compiler for IR lowering.

This document explains the project, the translation rules and the evidence, then
annotates the implementation line by line. The source commentary covers
`compile.py`, `emmy_ptx.py`, `tools/validate.py` and `tests/test_compiler.py`.
The original harness and optional fixture-generation utilities are explained by
their roles below; the generated IR strings, CUDA reference files and binary
NumPy fixtures are test data rather than implementation source.

Use these links to navigate the longer sections:

- [Compile and run](#compile-and-run)
- [Lowering rules and correctness details](#lowering-rules-and-correctness-details)
- [Validation evidence](#validation-evidence-from-october-22-2026)
- [Scope and limits](#scope-and-limits)
- [Line by line source commentary](#line-by-line-source-commentary)

## The problem and the resulting pipeline

Emmy's existing frontend and scheduler already decide what a GPU kernel should
compute and how threads cooperate. The usual route renders its Kernel IR as
CUDA C++, then uses nvcc to obtain a GPU binary. The backend added here starts
after scheduling:

```text
PyTorch expression
    |
    v
Emmy frontend and scheduling       Existing upstream stages
    |
    v
Typed Kernel IR plus meta.json
    |
    v
compile.py -> emmy_ptx.Compiler    This implementation
    |
    v
ASCII PTX 8.0 targeting sm_89
    |
    v
ptxas                            NVIDIA assembler and optimizer
    |
    v
Native GPU cubin -> CUDA Driver API -> output buffers
```

The compiler writes a program, not precomputed answers. Its lowering does not
inspect the problem directory's name, reference CUDA source, expected outputs,
or test seeds. Shapes come from graph tensors and axis extents; operations come
from typed statements; the entry name comes from `KernelOp.name`; positional
arguments come from the matching metadata entry. The same rules therefore apply
to new IRs that use the supported vocabulary.

The output has the native binary format produced by the standard toolchain.
Instruction selection, register allocation and floating-point evaluation order
can differ, so there is no claim that the cubin is byte-identical to nvcc's.
The README's approximately 19 times figure motivates the approach. The
[review and benchmark report](REVIEW_AND_BENCHMARK.md) measures the public nvcc
CUDA-reference-to-cubin path separately from direct single-file, API and worker
compilation; it does not time upstream IR-to-CUDA rendering.

## Repository map and requirements

| Path | Purpose |
| --- | --- |
| `README.md` | Setup, implementation status and original hackathon brief |
| `TASK.md` | Submission contract, IR reference and eligibility rule |
| `MVP.md` | Minimum requirement of all eight pointwise and reduction problems |
| `compile.py` | Required single-file command entry point |
| `emmy_ptx.py` | Original typed IR to PTX translation |
| `requirements.txt` | Emmy 0.3.11 and validation dependencies |
| `submit.py` | Original official runner client and output comparison |
| `tools/validate.py` | Local assembly, GPU execution and JSON reporting |
| `tools/benchmark.py` | Repeated direct/API/worker comparisons against public nvcc |
| `tests/test_compiler.py` | New synthetic IR and CLI regressions |
| `problems/` | Fifteen public problem packages |
| `examples/k_sum_1_reduce.ptx` | Starter repository's handwritten reduction example |
| `docs/API.md` | Official runner HTTP interface |
| `tools/expressions_open.py` | Original torch expressions and NumPy references |
| `tools/generate.py` | Optional organizer utility for generating new fixture packages |
| `docs/DEMO_VIDEO_SCRIPT.md` | Timed video script and recording commands |
| `build/` | Ignored PTX, cubin and validation artifacts |

Python 3.12 or newer is required by the repository. The implementation was
validated with Python 3.14.5, `emmy-ml==0.3.11`, CUDA Toolkit 13.4's ptxas,
and NumPy. Torch is not needed for loading or compiling existing IR. Generating
new IR from torch expressions is a separate optional upstream workflow.

`requirements.txt` line 1 pins the exact Emmy deserializer version. Lines 2 and 3
provide NumPy for expected-output checks and requests for the original HTTP
harness. The added `.gitignore` entries keep the local virtual environment,
download cache and generated artifacts out of submissions.

## Compile and run

From the repository root, with the intended Python environment active:

```bash
python -m pip install -r requirements.txt
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
ptxas -arch=sm_89 out.ptx -o out.cubin
python submit.py r01_array_sum --compiler compile.py
```

The first compile command satisfies the evaluator's exact interface. `meta.json`
must be next to the IR file, including when the file is moved into another
directory. The compiler can generate PTX on a machine with no CUDA toolkit or
GPU. The second command needs ptxas and produces the cubin explicitly. The third
command runs compilation and execution on the hosted GPU through the original
harness; it needs network access, not a local GPU.

On Windows, activate `.venv\Scripts\Activate.ps1`; on Linux, activate
`.venv/bin/activate`, if using a virtual environment. Use a shell that writes
ASCII-compatible redirected text. PowerShell 7 uses UTF-8 for this workflow;
older Windows PowerShell can write UTF-16 with `>`, so use `cmd /c` redirection
or explicitly write UTF-8 there.

For all local problems and the additional regressions:

```bash
python tools/validate.py
python tools/validate.py --assemble-only
python -m unittest discover -s tests -v
```

To execute the synthetic tests as well, set the environment variable before
running unittest. In PowerShell use `$env:EMMY_PTX_GPU_TESTS='1'`; in a POSIX
shell use `EMMY_PTX_GPU_TESTS=1 python -m unittest discover -s tests -v`.
The synthetic GPU cases require a device supporting `sm_89` and ptxas on PATH.
The default CPU test run generates PTX and checks CLI behavior but does not
assemble or execute the synthetic kernels.

## What is inside a problem

Each problem has a graph serialized in `*.ir.json`, a readable `*.ir.txt`, a
reference `*.ref.cu`, launch metadata, and an `expected.npz` archive. The compiler
reads only the JSON graph and metadata. The validator reads the expected output
separately after compilation to judge correctness.

At the graph level, input and constant nodes describe tensors with shapes and
types. A `KernelOp` contains a body of typed statements. The on-disk body contains
serialized representations that Emmy knows how to deserialize. The backend calls
`Graph.from_dict` rather than parsing those strings itself.

In `p02_scale_shift`, values such as 2.0 and 1.0 are one-element constant buffers.
They remain pointers in the kernel ABI and are loaded with ordinary memory
instructions. A numeric `Let` literal inside a kernel is different: it becomes
an immediate PTX operand. Confusing these two kinds of constant would break the
runtime parameter contract.

The matching `meta.json` entry supplies `arg_order`, argument descriptors, grid,
block, dynamic shared bytes and outputs that must start at zero. Its tolerances
define the comparison. It is not safe to reconstruct argument order from
`node.inputs`, even when that order happens to match in a particular fixture.

## Compiler state and register model

`Compiler` has a value environment for SSA names and loop variables, a separate
map for buffer pointers, and tables for buffer shapes and shared-memory bases.
Keeping pointers separate prevents a data-buffer name from colliding with a
compute-value name. Its output is assembled from buffered instruction strings,
function declarations and module declarations.

`Value(text, dtype)` represents either a virtual register or an immediate.
The four register families are 32-bit integer bit containers, 64-bit pointer
containers, float32 registers and predicates. Integer instructions select signed
or unsigned interpretation as needed; the `.b32` register declaration permits
both. Each allocation gets a new virtual name. Integer operands also retain safe
bounds for simplified index arithmetic. Named accumulators keep a stable
register so loop iterations and coupled combine programs update state in place.

All floating literals are encoded as `0fXXXXXXXX`, the exact IEEE float32 bit
pattern. This makes zero, negative identities and other constants independent of
decimal spelling or locale. Final PTX is encoded as ASCII to reject invalid
characters before anything is written to stdout.

## Lowering rules and correctness details

The instruction forms and address spaces follow NVIDIA's
[PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).
`ptxas` performs native instruction selection and physical register allocation.

| IR construct | Lowering in this backend |
| --- | --- |
| `Tile` | Linear thread ID, bounds branch and row-major coordinate decode |
| `Var`, `Literal` | Environment lookup or typed immediate |
| `BinaryExpr` | Recursive arithmetic or predicate instructions |
| `TernaryExpr` | Evaluate pure alternatives and select with `selp` |
| `Load`, `Write` | Flattened global/shared access, scalar or v2/v4 |
| `Assign`, `Let`, `Init` | Arithmetic, pure expression binding or explicit initialization |
| `Accum` | Operation on the current or explicitly rescaled base and incoming value |
| `Loop`, `StridedLoop` | Initialization, loop labels, comparison, body, increment and back edge |
| `Cond` | Predicate branch, then body, else body and join |
| `Smem` | Aligned static arrays or slices of the dynamic shared pool |
| `WarpShuffle` | Active-mask XOR butterfly followed by the supplied combine program |
| `TreeHalve` | Shared partials, pairwise combine, barrier per level and final broadcast |
| `Sync` | CTA/named barrier or active-mask warp barrier |
| `CpAsyncCopy` | Global-to-shared async copy with the IR's byte count |
| `CpAsyncCommit`, `CpAsyncWait` | Preserve the IR's async group boundaries |
| Atomic `Write` | Global float32 atomic add without a return value |

### Thread mapping and memory addresses

For a regular tile, `gid = blockIdx.x * blockDim.x + threadIdx.x`. A cooperative
tile uses `block_threads`, and auxiliary producer threads, when present, wrap
their local coordinate modulo the compute band. A branch excludes `gid >= N`,
where `N` is the product of tile-axis extents. Unrolled cells are already explicit
in the body; the backend does not confuse tile cells with original tensor cells.

For axis `i`, the coordinate is `(gid / product(inner_extents)) % extent_i`.
For a tensor shaped `(R, C)`, a load at `(row, col)` has element offset
`row * C + col`. The general recurrence is `flat = flat * dimension + coordinate`.
Multiplying by four gives a byte offset. Global addresses use a widened signed
32-bit element index and a 64-bit pointer addition; shared addresses use 32 bits.

Grouped loads and stores use v2 or v4 instructions. Correct alignment and
contiguous grouping follow the scheduled IR; the backend does not invent vector
groupings across arbitrary indices. The synthetic vector test checks a 7 by 20
tensor, four values per cell, 35 useful cells and 64 launched threads.

CTA raster hints only alter the order in which independent work tiles are
assigned to blocks. This backend visits the same cells in row-major order.
This preserves coverage and atomic accumulation semantics while giving up that
cache-locality optimization. Unroll annotations likewise leave runtime loops
with the same iteration semantics.

### Loop state and reductions

A seeded loop initializes accumulators belonging to that loop before its first
iteration. The binding scan descends through conditions but stops at nested loops; those
loops manage their own seeds. If `seed=False`, the earlier initialization remains
in force. This is essential for matmul, where each shared-memory slab contributes
to the same running output state, and for attention, where an inner dot product
resets for each outer key position.

Each loop also has its own value environment, so a locally declared name cannot
overwrite an outer binding. Only that loop's carried accumulators are exported
when it finishes. This fixes a shadowing case found in the code review.

`Accum.base` changes the left input to the update. For online softmax, this is a
previous sum multiplied by a scale derived from a new maximum. Replacing that
base with the plain accumulator would silently change the algorithm.

Each warp butterfly step first snapshots every remote state component, then
executes `combine_states` in order. It does not update a maximum before capturing
the other lane's sum. The tree uses the same generic combine program. For
warp-indexed shared trees, only lane zero writes a warp's slot, avoiding redundant
concurrent writers. All participating threads reach a barrier after each tree
level, and all then read the root result.

A final barrier protects those broadcast reads before shared scratch can be
reused by later statements in the same block.

For row softmax, the state consists of a pivot `m` and a normalized sum `s`.
Combining two partial states uses `m_new = max(m_a, m_b)` and
`s_new = exp(m_a - m_new) * s_a + exp(m_b - m_new) * s_b`.
The IR expands that relation into ordinary assignments. Reusing its combine
program also supports more state components without recognizing a softmax kernel
by name.

### Shared memory and asynchronous matmul

Shared declarations are collected before lowering the body, so conditional uses
can refer to bases initialized in the prologue. Each allocation starts at a
power-of-two-aligned offset. Up to 48 KiB uses static shared arrays; above that
threshold the module declares an external dynamic shared pool and the compiler
checks that the prescribed launch provides enough bytes.

The 512 by 512 matmul needs 73,728 bytes: the A slab begins at byte 0 and B at
24,576. The dynamic declaration belongs at module scope. The local host also
opts the function into the larger dynamic shared-memory limit before launching.

For unswizzled lane-based copies, destination coordinates are the slab base
coordinates plus the lane coordinates. A 16-byte copy uses `.cg`; four- and
eight-byte copies use `.ca`. Commit and wait operations remain in the exact IR
order, and the explicit synchronization statements provide CTA visibility before
the compute stage reads the slab.

The rectangular split-K matmul uses atomic adds. The compiler emits those adds;
the runtime initializes the named outputs to zero. Keeping the zeroing outside
the kernel avoids inter-block races and follows `zero_outputs` in metadata.

### Arithmetic and numerical behavior

Float32 addition, subtraction, multiplication, division and square root use
round-to-nearest instruction forms. Exponential uses
`exp(x) = 2 ** (x * log2(e))` with `ex2.approx`; logarithm uses `lg2.approx` and
a scale by `ln(2)`. The backend also handles copy, min/max, ReLU, reciprocal,
reciprocal square root, negation and absolute value. Some additional arithmetic
forms have lowering support beyond those exercised by the open set.

Floating-point reductions may differ from NumPy because addition is not
associative. Validation uses each problem's `rtol` and `atol`, as the official
harness does. Passing those checks is the correctness claim; numerical
equivalence for arbitrary NaNs, infinities or unsupported dtypes is not asserted.

## Validation evidence from October 22 2026

The official checks used the unmodified command
`python submit.py <problem> --compiler compile.py`. Its runner reported an
RTX 5090 and assembled for its default `sm_120`, accepting the PTX's `sm_89`
target. Local validation assembled `sm_89` cubins and ran them on an NVIDIA
RTX 3500 Ada Generation Laptop GPU.

| Category | Problem | Official result | Maximum absolute error |
| --- | --- | --- | --- |
| Pointwise | `p00_scalar_add` | PASS | 0 |
| Pointwise | `p01_vector_add` | PASS | 0 |
| Pointwise | `p02_scale_shift` | PASS | 0 |
| Pointwise | `p03_relu_mul` | PASS | 0 |
| Reduction | `r01_array_sum` | PASS | 3.815e-6 |
| Reduction | `r02_big_sum` | PASS | 1.526e-5 |
| Reduction | `r03_mean` | PASS | 9.313e-10 |
| Reduction | `r04_row_sum` | PASS | 5.722e-6 |
| Matmul | `m01_matmul_512` | PASS | 8.774e-5 |
| Matmul | `m02_matmul_rect` | PASS | 1.144e-5 |
| Fused | `f01_rmsnorm` | PASS | 1.907e-6 |
| Fused | `f02_row_softmax` | PASS | 7.451e-9 |
| Fused | `f03_l2norm` | PASS | 1.490e-8 |
| Attention | `a01_attention` | PASS | 6.557e-7 |
| Attention | `a02_attention_128` | PASS | 4.768e-7 |

All 15 also pass locally. The local JSON report records compilation and assembly
wall time, output error and assembler messages; these timings are diagnostic,
not a comparison with nvcc or a controlled GPU benchmark. Official raw output is
retained in `build/remote-validation.txt` in this working checkout. Running the
local validator regenerates `build/validation/report.json` and artifacts.

Fourteen unittest methods pass. GPU cases cover vector subtraction with reordered
arguments and a partly filled launch; maximum reduction with empty thread
partials; nested-loop accumulator preservation and local shadowing; signed index
division and modulo; predicate expressions; and declared literal conversion.
Other checks cover CLI/worker error handling, reduction/stride/type validation,
unsupported operations and signed-zero literal encoding. These tests do not
access the hidden evaluation set.

## Scope and limits

The implementation targets static float32 kernels and one-dimensional launches
with 32-bit element indexing and 64-bit global pointers. It supports the
non-tensor-core statement forms exercised by the open problems. There is no
general optimization pipeline or liveness analysis.

By-value scalar arguments, non-f32 data buffers, dynamic shapes, serial runtime
axes, software shared-memory swizzles, TMA, carried arrays and tensor-core
fragments are not implemented. Unsupported statements and arithmetic operations
produce an error before PTX is written. The backend assumes a well-formed,
scheduled IR: valid cooperative participation, aligned vectors, positive strides
and indices that fit the supported integer width. Named/warp barriers,
auxiliary threads and segmented shared trees have translation paths but are
not exercised by the 15 public problems or the added GPU tests.

Passing the public fixtures demonstrates the supported compilation path. Hidden
cases can still reveal unsupported operations, schedules or numerical edge
cases. The video is still to be recorded by a team member, and repository
publication and hackathon submission remain user actions.

## How the existing official harness works

`submit.py` reads a selected problem, loads its metadata and expected arrays,
then runs the compiler as a subprocess once per kernel. It captures stdout,
rejects empty or obviously invalid PTX, posts the source to `/compile` and
checks the returned entry points. It then constructs positional arguments from
the metadata, chaining any prior outputs, and posts the cubin handle and launch
to `/run`. A busy runner is retried. Returned buffers are reshaped to the NumPy
reference shapes and compared with `np.allclose`; any failed output makes the
process exit unsuccessfully. This script is unchanged by the implementation.

`tools/expressions_open.py` is a list of problem specifications, each containing a
torch expression, NumPy reference function, tolerance and optional scheduling
knobs. `tools/generate.py` traces those expressions through upstream Emmy and
packages their IR, readable body, CUDA reference and metadata. It is a fixture
generation tool; the new compiler never invokes it. In particular, its use of
upstream CUDA generation does not form part of this backend's lowering path.

## Line by line source commentary

The sections below reproduce each implementation source file with verified
one-based line numbers and explain every nonblank line. Blank lines only separate
Python definitions or logical steps. A line that closes a tuple, dictionary or
multiline call completes the operation described immediately before it.
Line numbers and hashes identify this source snapshot; update this commentary
when changing the implementation.

<!-- SOURCE_COMMENTARY -->

### compile.py

Source snapshot SHA256 `315927f1465bc2b5bb7d11ca234892ca6b3863d61b5fdf46388ffa70d9672ca9`.

#### Module imports and definitions at lines 1 to 11

```python
   1 | #!/usr/bin/env python3
   2 | """Submission entry point: python compile.py kernel.ir.json > kernel.ptx."""
   3 | 
   4 | import argparse
   5 | import json
   6 | import sys
   7 | from pathlib import Path
   8 | 
   9 | from emmy_ptx import CompileError, compile_file
  10 | 
  11 | 
```

| Line | Explanation |
| --- | --- |
| 1 | Permit direct execution on systems that honor the Python shebang. |
| 2 | State the evaluator's required positional-path and stdout contract. |
| 4 | Import argument parsing for the single input-file interface. |
| 5 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 6 | Import process exit and stdout/stderr access. |
| 7 | Represent the input as a platform-correct filesystem path. |
| 9 | Import only the public compilation function and diagnostic exception from the backend. |

#### worker at lines 12 to 25

Retain imports and compile one freshly read IR per JSON-line request with recoverable errors.

```python
  12 | def worker() -> int:
  13 |     """Compile JSON-line requests in one process, amortizing Emmy imports."""
  14 |     for line in sys.stdin:
  15 |         try:
  16 |             request = json.loads(line)
  17 |             if not isinstance(request, dict) or not isinstance(request.get("ir"), str):
  18 |                 raise ValueError('request must contain a string "ir" path')
  19 |             response = {"ok": True, "ptx": compile_file(Path(request["ir"]))}
  20 |         except (CompileError, OSError, ValueError, KeyError, TypeError) as exc:
  21 |             response = {"ok": False, "error": str(exc)}
  22 |         print(json.dumps(response), flush=True)
  23 |     return 0
  24 | 
  25 | 
```

| Line | Explanation |
| --- | --- |
| 12 | Retain imports and compile one freshly read IR per JSON-line request with recoverable errors. |
| 13 | Describe the purpose or usage of this definition. |
| 14 | Process requests sequentially while retaining loaded dependencies in one process. |
| 15 | Enter an operation whose failure or cleanup is handled by the following clauses. |
| 16 | Parse exactly one JSON object from the request line. |
| 17 | Require an object containing a string filesystem path. |
| 18 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 19 | Return complete generated PTX in the successful response object. |
| 20 | Convert the listed expected exceptions into the error behavior described by this function. |
| 21 | Return a structured diagnostic while allowing later requests to continue. |
| 22 | Send one response line immediately so the caller can proceed without a buffering delay. |
| 23 | Return the completed result to the caller of this function. |

#### main at lines 26 to 47

```python
  26 | def main() -> int:
  27 |     parser = argparse.ArgumentParser(description=__doc__)
  28 |     parser.add_argument("ir", type=Path, nargs="?")
  29 |     parser.add_argument("--worker", action="store_true", help="serve JSON-line compile requests on stdin/stdout")
  30 |     args = parser.parse_args()
  31 |     if args.worker:
  32 |         if args.ir is not None:
  33 |             parser.error("--worker cannot be combined with an IR path")
  34 |         return worker()
  35 |     if args.ir is None:
  36 |         parser.error("an IR path is required")
  37 |     try:
  38 |         ptx = compile_file(args.ir)
  39 |     except (CompileError, OSError, ValueError, KeyError, TypeError) as exc:
  40 |         print(f"compile.py: {exc}", file=sys.stderr)
  41 |         return 1
  42 |     sys.stdout.write(ptx)
  43 |     return 0
  44 | 
  45 | 
  46 | if __name__ == "__main__":
  47 |     sys.exit(main())
```

| Line | Explanation |
| --- | --- |
| 26 | Define a CLI function that reports success or failure through an exit code. |
| 27 | Use the module contract as help text. |
| 28 | Allow the positional path to be absent only for the explicitly selected worker mode. |
| 29 | Expose persistent JSON-line compilation without changing single-file stdout. |
| 30 | Parse argv while retaining argparse's standard usage errors. |
| 31 | Select the explicit JSON protocol instead of the default raw-PTX protocol. |
| 32 | Reject combining a worker session with a positional one-shot input. |
| 33 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |
| 34 | Serve requests until stdin closes and return a clean exit status. |
| 35 | Keep the submission mode's input path mandatory when worker mode is absent. |
| 36 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |
| 37 | Handle expected input, deserialization and lowering failures before writing output. |
| 38 | Build the entire PTX module in memory. |
| 39 | Catch user-input and supported compiler diagnostics at the command boundary. |
| 40 | Write a concise error to stderr, keeping stdout free of non-PTX text. |
| 41 | Return a nonzero process status when compilation fails. |
| 42 | Write the completed PTX unchanged to stdout for shell redirection. |
| 43 | Return success only after successful compilation and output. |
| 46 | Run the CLI only when this file is executed as the program. |
| 47 | Convert main's result into the process exit status used by the evaluator. |

### emmy_ptx.py

Source snapshot SHA256 `aac65241768e2cd188ddf2e91712000becf0a4a33247ec897e811eea8ec150d1`.

#### Module imports and definitions at lines 1 to 33

```python
   1 | """A direct, scalar f32 Kernel IR to PTX backend. No CUDA source is generated."""
   2 | 
   3 | from __future__ import annotations
   4 | 
   5 | import json
   6 | import math
   7 | import operator
   8 | import re
   9 | import struct
  10 | from collections import Counter
  11 | from dataclasses import dataclass
  12 | from functools import lru_cache
  13 | from pathlib import Path
  14 | 
  15 | from emmy.compiler.graph import Graph
  16 | from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
  17 | from emmy.compiler.ir.kernel.ir import (
  18 |     CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Smem, Sync,
  19 |     Tile, TreeHalve, WarpShuffle,
  20 | )
  21 | from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write
  22 | 
  23 | REGISTER_PREFIX = {"s32": "r", "f32": "f", "u64": "rd", "pred": "p"}
  24 | MOVE_TYPES = {"s32": "b32", "f32": "f32", "u64": "b64", "pred": "pred"}
  25 | COMPARISONS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
  26 | INTEGER_OPS = {"+": "add", "-": "sub", "*": "mul.lo", "^": "xor", "&": "and", "|": "or", ">>": "shr", "<<": "shl"}
  27 | FLOAT_OPS = {"*": "mul.rn", "+": "add.rn", "-": "sub.rn", "/": "div.rn"}
  28 | ELEMENTWISE_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
  29 | INTEGER_FOLD = {"+": operator.add, "-": operator.sub, "*": operator.mul,
  30 |                 "/": operator.floordiv, "//": operator.floordiv, "%": operator.mod,
  31 |                 "&": operator.and_, "|": operator.or_, "^": operator.xor}
  32 | 
  33 | 
```

| Line | Explanation |
| --- | --- |
| 1 | Document the purpose and usage of this module, type or method. |
| 3 | Import the required definitions from `__future__`; no renderer is invoked here. |
| 5 | Import `json` for the module operations described above. |
| 6 | Import `math` for the module operations described above. |
| 7 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 8 | Import `re` for the module operations described above. |
| 9 | Import `struct` for the module operations described above. |
| 10 | Import the required definitions from `collections`; no renderer is invoked here. |
| 11 | Import the required definitions from `dataclasses`; no renderer is invoked here. |
| 12 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 13 | Import the required definitions from `pathlib`; no renderer is invoked here. |
| 15 | Import the required definitions from `emmy.compiler.graph`; no renderer is invoked here. |
| 16 | Import the required definitions from `emmy.compiler.ir.expr`; no renderer is invoked here. |
| 17 | Import the required definitions from `emmy.compiler.ir.kernel.ir`; no renderer is invoked here. |
| 18 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 19 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 20 | Complete the multiline structure or call that begins on the preceding lines. |
| 21 | Import the required definitions from `emmy.compiler.ir.stmt`; no renderer is invoked here. |
| 23 | Share the register namespace lookup table across lowering calls. |
| 24 | Share the legal typed-move suffix lookup table. |
| 25 | Share the mapping from IR comparison spellings to PTX comparison qualifiers. |
| 26 | Share integer instruction stems rather than rebuilding a mapping on every expression. |
| 27 | Share the float32 opcode and nearest-rounding policy. |
| 28 | Map registered scalar operation names to the backend arithmetic rules. |
| 29 | Use ordinary Python integer operations solely to fold literal arithmetic in this backend. |
| 30 | Compute and bind `INTEGER_FOLD` for the surrounding operation described in this function. |
| 31 | Compute and bind `INTEGER_FOLD` for the surrounding operation described in this function. |

#### CompileError at lines 34 to 37

```python
  34 | class CompileError(ValueError):
  35 |     """An input requires a feature this backend cannot lower faithfully."""
  36 | 
  37 | 
```

| Line | Explanation |
| --- | --- |
| 34 | Introduce the class that groups the following state and behavior. |
| 35 | Document the purpose and usage of this module, type or method. |

#### symbol at lines 38 to 43

```python
  38 | def symbol(name: str) -> str:
  39 |     if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name, flags=re.ASCII):
  40 |         raise CompileError(f"invalid PTX symbol {name!r}")
  41 |     return name
  42 | 
  43 | 
```

| Line | Explanation |
| --- | --- |
| 38 | Define `symbol` with the parameters and result contract shown in this signature. |
| 39 | Constrain emitted identifiers to the ASCII subset accepted by this backend. |
| 40 | Reject an invalid symbol before creating PTX that ptxas cannot parse. |
| 41 | Return the validated identifier unchanged. |

#### walk at lines 44 to 51

```python
  44 | def walk(body):
  45 |     for stmt in body:
  46 |         yield stmt
  47 |         for child in stmt.nested():
  48 |             yield from walk(child)
  49 | 
  50 | 
  51 | @dataclass(frozen=True, slots=True)
```

| Line | Explanation |
| --- | --- |
| 44 | Define `walk` with the parameters and result contract shown in this signature. |
| 45 | Traverse statements in their original execution order. |
| 46 | Expose the current statement to the shared-memory collection pass. |
| 47 | Ask the typed statement for its child bodies, rather than parsing serialized text. |
| 48 | Recurse into every child to discover declarations inside loops and branches. |
| 51 | Keep operands immutable and use slots to reduce per-operand storage and allocation overhead. |

#### Value at lines 52 to 59

```python
  52 | class Value:
  53 |     text: str
  54 |     dtype: str
  55 |     integer: int | None = None
  56 |     minimum: int | None = None
  57 |     maximum: int | None = None
  58 | 
  59 | 
```

| Line | Explanation |
| --- | --- |
| 52 | Introduce the class that groups the following state and behavior. |
| 53 | Store the PTX spelling of a register or literal. |
| 54 | Store the semantic type used to choose opcodes and register families. |
| 55 | Record the exact value only for literal integers, allowing safe integer constant folding. |
| 56 | Track a proven signed lower bound for an integer operand, or unknown. |
| 57 | Track a proven signed upper bound paired with the lower bound. |

#### constant at lines 60 to 70

```python
  60 | def constant(value, dtype="f32") -> Value:
  61 |     if dtype == "f32":
  62 |         bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
  63 |         return encoded_constant(bits, dtype)
  64 |     number = int(value)
  65 |     if not -(1 << 31) <= number < (1 << 31):
  66 |         raise CompileError("integer literal exceeds signed 32-bit indexing")
  67 |     return encoded_constant(number, dtype)
  68 | 
  69 | 
  70 | @lru_cache(maxsize=1024)
```

| Line | Explanation |
| --- | --- |
| 60 | Define `constant` and the input/output contract shown in its signature. |
| 61 | Check the stated condition before executing the following guarded operation. |
| 62 | Compute and bind `bits` for the surrounding operation described in this function. |
| 63 | Cache the encoded bit pattern, preserving the distinction between positive and negative zero. |
| 64 | Normalize an integer literal before checking the backend indexing width. |
| 65 | Reject a literal that cannot be represented as a signed 32-bit index. |
| 66 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 67 | Reuse a bounded cached operand after validating the integer value. |
| 70 | Bound literal reuse to 1024 encoded operands for a long-running compiler worker. |

#### encoded_constant at lines 71 to 77

```python
  71 | def encoded_constant(number, dtype):
  72 |     # Cache bit patterns, since Python considers +0.0 and -0.0 equal as keys.
  73 |     if dtype == "f32":
  74 |         return Value(f"0f{number:08X}", dtype)
  75 |     return Value(str(number), dtype, number, number, number)
  76 | 
  77 | 
```

| Line | Explanation |
| --- | --- |
| 71 | Define `encoded_constant` and the input/output contract shown in its signature. |
| 72 | Document the adjacent implementation rule or invocation. |
| 73 | Check the stated condition before executing the following guarded operation. |
| 74 | Spell an exact float32 immediate from its unsigned IEEE bit pattern. |
| 75 | Record an integer literal with an exact one-value interval. |

#### Compiler at lines 78 to 78

```python
  78 | class Compiler:
```

| Line | Explanation |
| --- | --- |
| 78 | Introduce the class that groups the following state and behavior. |

#### __init__ at lines 79 to 122

```python
  79 |     def __init__(self, graph: Graph, node, launch: dict):
  80 |         self.kernel = node.op
  81 |         self.launch = launch
  82 |         self.lines: list[str] = []
  83 |         self.counts: Counter = Counter()
  84 |         self.labels = 0
  85 |         self.env: dict[str, Value] = {}
  86 |         self.buffers = {}
  87 |         self.pointers = {}
  88 |         self.shared = {}
  89 |         self.declarations = []
  90 |         self.module_declarations = []
  91 |         for n in graph.nodes.values():
  92 |             for tensor in n.outputs:
  93 |                 self.buffers[tensor.name] = (tuple(d.as_static() for d in tensor.shape), tensor.dtype.name)
  94 |         if self.kernel.serial:
  95 |             raise CompileError("serial runtime axes are not supported")
  96 |         if launch["block"][1:] != [1, 1] or launch["grid"][1:] != [1, 1]:
  97 |             raise CompileError("only the Kernel IR's one dimensional launch is supported")
  98 |         self.params = launch["arg_order"]
  99 |         if len(set(self.params)) != len(self.params):
 100 |             raise CompileError("duplicate names in arg_order")
 101 |         args = {arg["name"]: arg for arg in launch["args"]}
 102 |         if set(args) != set(self.params) or not set(node.inputs).issubset(args):
 103 |             raise CompileError("metadata arguments do not match the kernel inputs")
 104 |         if not {t.name for t in node.outputs}.issubset(args):
 105 |             raise CompileError("metadata is missing a kernel output")
 106 |         for i, name in enumerate(self.params):
 107 |             symbol(name)
 108 |             if args[name].get("scalar"):
 109 |                 raise CompileError("by-value scalar parameters are not supported; constants must be buffers")
 110 |             if args[name]["dtype"] != "f32" or self.buffers.get(name, (None, None))[1] != "f32":
 111 |                 raise CompileError(f"buffer {name}: this backend supports f32 data only")
 112 |             pointer = self.reg("u64")
 113 |             self.emit(f"ld.param.u64 {pointer.text}, [param_{i}];")
 114 |             self.emit(f"cvta.to.global.u64 {pointer.text}, {pointer.text};")
 115 |             self.pointers[name] = pointer
 116 |         self.prepare_shared()
 117 |         self.tid = self.instruction("s32", "mov.u32", "%tid.x", bounds=(0, launch["block"][0]-1))
 118 |         self.env["lane"] = self.instruction("s32", "and.b32", self.tid, 31)
 119 |         self.env["warp"] = self.instruction("s32", "shr.u32", self.tid, 5)
 120 |         self.bid = self.instruction("s32", "mov.u32", "%ctaid.x")
 121 |         self.ntid = self.instruction("s32", "mov.u32", "%ntid.x")
 122 | 
```

| Line | Explanation |
| --- | --- |
| 79 | Define `__init__` with the parameters and result contract shown in this signature. |
| 80 | Retain the typed KernelOp rather than its graph wrapper. |
| 81 | Retain the matching launch metadata for ABI and shared-memory checks. |
| 82 | Buffer instructions until register counts and declarations are known. |
| 83 | Track how many registers have been allocated in each family. |
| 84 | Start the counter used to avoid duplicate branch labels. |
| 85 | Map compute names and induction variables to stable typed operands. |
| 86 | Map buffer names to their static shape and element type. |
| 87 | Keep global buffer pointers in a namespace separate from scalar compute values. |
| 88 | Map shared buffer names to 32-bit shared-memory addresses. |
| 89 | Collect declarations that belong inside the kernel function. |
| 90 | Collect declarations, such as external shared storage, that belong at module scope. |
| 91 | Visit all graph nodes because parameters can include constants and intermediates. |
| 92 | Inspect every tensor produced by a graph node. |
| 93 | Require static dimensions and record each tensor's shape and dtype. |
| 94 | Detect the separate runtime serial-axis ABI, which is outside this backend's scope. |
| 95 | Fail explicitly instead of emitting an incomplete kernel signature. |
| 96 | Check that both grid and block use only their x dimensions. |
| 97 | Reject a launch that would require a different thread-coordinate calculation. |
| 98 | Use the metadata's positional argument contract, independent of graph input order. |
| 99 | Check for ambiguous duplicate ABI slots named as the same buffer. |
| 100 | Report the invalid argument order before emitting a signature. |
| 101 | Index argument descriptors by name for validation while preserving arg_order separately. |
| 102 | Require descriptor coverage and inclusion of all kernel inputs. |
| 103 | Explain a mismatch between the graph inputs and metadata. |
| 104 | Ensure every kernel output also has a runtime argument descriptor. |
| 105 | Reject an ABI that could not provide storage for an output. |
| 106 | Assign parameter slot numbers strictly in arg_order order. |
| 107 | Validate each buffer identifier before using it in generated declarations or diagnostics. |
| 108 | Distinguish by-value scalars from the one-element pointer buffers used for constants. |
| 109 | Reject the unsupported scalar calling convention explicitly. |
| 110 | Cross-check graph and metadata element types against the implemented f32 format. |
| 111 | Reject unsupported data widths before calculating any addresses. |
| 112 | Allocate a 64-bit virtual register for a global pointer. |
| 113 | Read the pointer from the corresponding positional parameter slot. |
| 114 | Convert the generic parameter address to a global-space pointer. |
| 115 | Keep the pointer in its separate buffer namespace so compute bindings cannot overwrite it. |
| 116 | Collect shared allocations and emit their base-pointer initialization. |
| 117 | Read the local thread index and retain the launch-proven range from zero to block size minus one. |
| 118 | Compute the lane index as thread index modulo 32. |
| 119 | Compute the warp index as thread index divided by 32. |
| 120 | Read the x block index used to form the global work index. |
| 121 | Read the actual launch's x block size, rather than guessing a default. |

#### emit at lines 123 to 125

```python
 123 |     def emit(self, line: str):
 124 |         self.lines.append("    " + line)
 125 | 
```

| Line | Explanation |
| --- | --- |
| 123 | Define `emit` with the parameters and result contract shown in this signature. |
| 124 | Indent and append one PTX line to the buffered function body. |

#### reg at lines 126 to 130

```python
 126 |     def reg(self, dtype: str, bounds=(None, None)) -> Value:
 127 |         prefix = REGISTER_PREFIX[dtype]
 128 |         self.counts[prefix] += 1
 129 |         return Value(f"%{prefix}{self.counts[prefix]}", dtype, None, *bounds)
 130 | 
```

| Line | Explanation |
| --- | --- |
| 126 | Define `reg` and the input/output contract shown in its signature. |
| 127 | Look up the precomputed register family for the requested type. |
| 128 | Reserve the next virtual register number in that namespace. |
| 129 | Create a virtual register carrying optional safe integer bounds and no literal value. |

#### label at lines 131 to 135

```python
 131 |     def label(self, prefix="L") -> str:
 132 |         self.labels += 1
 133 |         return f"{prefix}_{self.labels}"
 134 | 
 135 |     @staticmethod
```

| Line | Explanation |
| --- | --- |
| 131 | Define `label` with the parameters and result contract shown in this signature. |
| 132 | Advance the label counter for the whole kernel, including nested control flow. |
| 133 | Combine a readable label prefix with a unique suffix. |
| 135 | This formatter does not depend on any Compiler instance state. |

#### operand at lines 136 to 138

```python
 136 |     def operand(value) -> str:
 137 |         return value.text if isinstance(value, Value) else str(value)
 138 | 
```

| Line | Explanation |
| --- | --- |
| 136 | Define `operand` with the parameters and result contract shown in this signature. |
| 137 | Extract Value spelling or format a built-in register name or integer operand. |

#### instruction at lines 139 to 147

```python
 139 |     def instruction(self, dtype, opcode, *args, bounds=(None, None)) -> Value:
 140 |         result = self.reg(dtype, bounds)
 141 |         operands = ", ".join(self.operand(a) for a in args)
 142 |         tail = f", {operands}" if operands else ""
 143 |         self.emit(f"{opcode} {result.text}{tail};")
 144 |         return result
 145 | 
 146 |     constant = staticmethod(constant)
 147 | 
```

| Line | Explanation |
| --- | --- |
| 139 | Define `instruction` and the input/output contract shown in its signature. |
| 140 | Allocate the instruction result with its proven optional interval. |
| 141 | Format source operands in PTX comma-separated order. |
| 142 | Avoid a dangling comma for zero-source instructions such as activemask. |
| 143 | Append one complete instruction with a terminating semicolon. |
| 144 | Make the produced operand available to subsequent lowering steps. |
| 146 | Reuse the module-level literal encoder without retaining compiler instances in its cache. |

#### bind at lines 148 to 157

```python
 148 |     def bind(self, name, value: Value):
 149 |         if name not in self.env:
 150 |             self.env[name] = self.reg(value.dtype)
 151 |         dest = self.env[name]
 152 |         if dest.dtype != value.dtype:
 153 |             raise CompileError(f"inconsistent types for {name}: {dest.dtype} and {value.dtype}")
 154 |         mov_type = MOVE_TYPES[value.dtype]
 155 |         self.emit(f"mov.{mov_type} {dest.text}, {value.text};")
 156 |         self.env[name] = Value(dest.text, dest.dtype, None, value.minimum, value.maximum)
 157 | 
```

| Line | Explanation |
| --- | --- |
| 148 | Define `bind` with the parameters and result contract shown in this signature. |
| 149 | Allocate a named destination on its first definition only. |
| 150 | Reserve a stable register so future accumulator updates modify the same location. |
| 151 | Retrieve the existing or newly created destination binding. |
| 152 | Check that a name is never reused with an incompatible register type. |
| 153 | Report the two conflicting types at the offending IR value. |
| 154 | Select a legal move suffix from the shared type table. |
| 155 | Copy the expression result into the stable named destination. |
| 156 | Keep the stable named register but refresh its interval metadata after this assignment. |

#### expr at lines 158 to 186

```python
 158 |     def expr(self, expr, cache=None) -> Value:
 159 |         # Share only within this pure expression, never across assignments or loop edges.
 160 |         if cache is None:
 161 |             cache = {}
 162 |         if isinstance(expr, Var):
 163 |             if expr.name not in self.env:
 164 |                 raise CompileError(f"unbound IR value {expr.name!r}")
 165 |             return self.env[expr.name]
 166 |         if isinstance(expr, Literal):
 167 |             return self.constant(expr.value, "s32" if expr.dtype in ("int", "i32", "bool") else "f32")
 168 |         if isinstance(expr, BinaryExpr):
 169 |             left, right = self.expr(expr.left, cache), self.expr(expr.right, cache)
 170 |             key = (expr.op, left, right)
 171 |             if key not in cache:
 172 |                 cache[key] = self.binary(expr.op, left, right)
 173 |             return cache[key]
 174 |         if isinstance(expr, TernaryExpr):
 175 |             pred = self.predicate(self.expr(expr.cond, cache))
 176 |             yes, no = self.expr(expr.if_true, cache), self.expr(expr.if_false, cache)
 177 |             if yes.dtype != no.dtype:
 178 |                 raise CompileError("ternary branches have different types")
 179 |             if yes.dtype == "pred":
 180 |                 a = self.instruction("pred", "and.pred", pred, yes)
 181 |                 inverse = self.instruction("pred", "not.pred", pred)
 182 |                 b = self.instruction("pred", "and.pred", inverse, no)
 183 |                 return self.instruction("pred", "or.pred", a, b)
 184 |             return self.instruction(yes.dtype, f"selp.{yes.dtype}", yes, no, pred)
 185 |         raise CompileError(f"unsupported expression {type(expr).__name__}")
 186 | 
```

| Line | Explanation |
| --- | --- |
| 158 | Define `expr` and the input/output contract shown in its signature. |
| 159 | Document the adjacent implementation rule or invocation. |
| 160 | Start a cache for one pure expression evaluation only. |
| 161 | Allocate the expression-local reuse table without sharing it across statements. |
| 162 | Recognize a reference to an axis, accumulator or computed SSA name. |
| 163 | Detect a reference before any binding was established. |
| 164 | Name the missing value instead of emitting a bogus register operand. |
| 165 | Return the already established typed operand. |
| 166 | Recognize a literal expression rather than a buffer constant. |
| 167 | Preserve integer index literals and encode other numeric literals as float32. |
| 168 | Recognize a binary expression tree. |
| 169 | Recursively lower both arguments in the same pure-expression scope. |
| 170 | Identify a repeated operation by its operator and exact typed operands. |
| 171 | Emit the pure operation only once within this expression. |
| 172 | Remember the already generated result while its inputs cannot be reassigned. |
| 173 | Reuse the existing virtual result instead of emitting a duplicate instruction. |
| 174 | Recognize a pure conditional expression, distinct from a statement-level Cond. |
| 175 | Lower the ternary condition with the same expression-local reuse scope. |
| 176 | Evaluate both pure alternatives before selecting their result. |
| 177 | Require matching branch types because selp has one result type. |
| 178 | Reject an implicit conversion the backend has not implemented. |
| 179 | Use predicate logic because PTX selp does not have a predicate result type. |
| 180 | Keep the true alternative only when the condition holds. |
| 181 | Invert the condition for the false alternative. |
| 182 | Keep the false alternative only when the condition does not hold. |
| 183 | Merge the two mutually selected predicate alternatives. |
| 184 | Select the true or false operand with a typed selp instruction. |
| 185 | Reject an unknown expression node rather than attempting text substitution. |

#### predicate at lines 187 to 191

```python
 187 |     def predicate(self, value: Value) -> Value:
 188 |         if value.dtype == "pred":
 189 |             return value
 190 |         return self.instruction("pred", f"setp.ne.{value.dtype}", value, self.constant(0, value.dtype))
 191 | 
```

| Line | Explanation |
| --- | --- |
| 187 | Define `predicate` with the parameters and result contract shown in this signature. |
| 188 | An existing predicate can be used directly by a branch or logical operation. |
| 189 | Avoid generating a redundant comparison for predicate inputs. |
| 190 | Treat a nonzero numeric value as true by comparing with a same-type zero. |

#### binary at lines 192 to 225

```python
 192 |     def binary(self, op, left: Value, right: Value) -> Value:
 193 |         if op in ("&&", "||"):
 194 |             return self.instruction("pred", "and.pred" if op == "&&" else "or.pred",
 195 |                                     self.predicate(left), self.predicate(right))
 196 |         if left.dtype != right.dtype:
 197 |             raise CompileError(f"mixed expression types: {left.dtype}, {right.dtype}")
 198 |         if left.dtype == "pred":
 199 |             if op not in ("==", "!="):
 200 |                 raise CompileError(f"unsupported predicate operator {op}")
 201 |             different = self.instruction("pred", "xor.pred", left, right)
 202 |             return self.instruction("pred", "not.pred", different) if op == "==" else different
 203 |         if op in COMPARISONS:
 204 |             return self.instruction("pred", f"setp.{COMPARISONS[op]}.{left.dtype}", left, right)
 205 |         dtype = left.dtype
 206 |         if dtype == "f32":
 207 |             code = FLOAT_OPS.get(op)
 208 |             if code is None:
 209 |                 raise CompileError(f"unsupported float operator {op}")
 210 |         elif dtype == "s32":
 211 |             folded = self.fold_integer(op, left, right)
 212 |             if folded is not None:
 213 |                 return folded
 214 |             if op in ("/", "//", "%"):
 215 |                 return self.integer_divmod(op, left, right)
 216 |             code = INTEGER_OPS.get(op)
 217 |             if code is None:
 218 |                 raise CompileError(f"unsupported binary operator {op}")
 219 |         else:
 220 |             raise CompileError(f"unsupported arithmetic type {dtype}")
 221 |         suffix = "b32" if op in ("^", "&", "|", "<<") else dtype
 222 |         bounds = self.integer_bounds(op, left, right) if dtype == "s32" else (None, None)
 223 |         return self.instruction(dtype, f"{code}.{suffix}", left, right, bounds=bounds)
 224 | 
 225 |     @staticmethod
```

| Line | Explanation |
| --- | --- |
| 192 | Define `binary` with the parameters and result contract shown in this signature. |
| 193 | Handle logical conjunction and disjunction as predicate operations. |
| 194 | Select the predicate AND or OR instruction. |
| 195 | Convert both arguments to predicates before combining them. |
| 196 | Prevent implicit mixed-type arithmetic and comparison. |
| 197 | Report the actual operand type mismatch. |
| 198 | Separate predicate equality from numeric setp instructions. |
| 199 | Limit direct predicate comparisons to the implemented equality forms. |
| 200 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 201 | Predicate XOR is true exactly when the two predicate values differ. |
| 202 | Invert XOR for equality or keep it for inequality. |
| 203 | Handle numeric comparisons before selecting an arithmetic result type. |
| 204 | Return the completed result to the caller of this function. |
| 205 | Use the already checked common operand type to select arithmetic rules. |
| 206 | Floating-point operations require explicit rounding qualifiers instead of integer forms. |
| 207 | Choose the shared float32 operation and rounding form. |
| 208 | Detect an operator valid only for integers in this mapping. |
| 209 | Reject the unsupported float variant rather than choosing a misleading instruction. |
| 210 | Apply integer-only simplifications without changing floating-point evaluation. |
| 211 | Try literal folding and integer algebraic identities before allocating a result. |
| 212 | Avoid emitting an instruction when an exact operand already represents the answer. |
| 213 | Return the integer literal or unchanged operand selected by the simplifier. |
| 214 | Route signed division and modulo through the floor-semantics correction. |
| 215 | Preserve Emmy floor-division semantics for negative intermediate coordinates. |
| 216 | Select the shared integer instruction mapping for the remaining operators. |
| 217 | Check the stated condition before executing the following guarded operation. |
| 218 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 219 | Use the alternate path when the preceding condition does not hold. |
| 220 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 221 | Use bit-container suffixes for bitwise operations and typed suffixes otherwise. |
| 222 | Infer conservative result bounds for integer arithmetic, leaving float bounds unknown. |
| 223 | Emit the operation with interval metadata used only for later safe integer simplification. |
| 225 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |

#### integer_bounds at lines 226 to 235

Infer conservative integer intervals and discard any proof that would cross signed overflow.

```python
 226 |     def integer_bounds(op, left, right):
 227 |         if None in (left.minimum, left.maximum, right.minimum, right.maximum):
 228 |             return None, None
 229 |         if op not in ("+", "-", "*"):
 230 |             return None, None
 231 |         operation = INTEGER_FOLD[op]
 232 |         values = [operation(a, b) for a in (left.minimum, left.maximum) for b in (right.minimum, right.maximum)]
 233 |         lo, hi = min(values), max(values)
 234 |         return (lo, hi) if -(1 << 31) <= lo <= hi < (1 << 31) else (None, None)
 235 | 
```

| Line | Explanation |
| --- | --- |
| 226 | Infer conservative integer intervals and discard any proof that would cross signed overflow. |
| 227 | Stop range inference when either input lacks a complete known interval. |
| 228 | Represent an unproven interval explicitly rather than assuming a nonnegative value. |
| 229 | Infer arithmetic intervals only for the implemented addition, subtraction and multiplication rules. |
| 230 | Represent an unproven interval explicitly rather than assuming a nonnegative value. |
| 231 | Use the matching integer function to evaluate interval endpoints. |
| 232 | Evaluate all endpoint pairs so negative multiplication and subtraction remain conservative. |
| 233 | Take the enclosing interval of all endpoint results. |
| 234 | Discard inferred bounds when a signed 32-bit overflow could invalidate them. |

#### fold_integer at lines 236 to 252

Fold literal integer expressions and remove safe integer identities, leaving floating operations unchanged.

```python
 236 |     def fold_integer(self, op, left, right):
 237 |         a, b = left.integer, right.integer
 238 |         if op in ("/", "//", "%") and b == 0:
 239 |             raise CompileError("integer division by zero")
 240 |         if a is not None and b is not None:
 241 |             if op in INTEGER_FOLD:
 242 |                 value = INTEGER_FOLD[op](a, b)
 243 |                 value = (value + (1 << 31)) % (1 << 32) - (1 << 31)
 244 |                 return self.constant(value, "s32")
 245 |         if op == "+" and a == 0 or op == "*" and a == 1:
 246 |             return right
 247 |         if op in ("+", "-") and b == 0 or op == "*" and b == 1 or op in ("/", "//") and b == 1:
 248 |             return left
 249 |         if op == "*" and (a == 0 or b == 0) or op == "%" and b == 1:
 250 |             return self.constant(0, "s32")
 251 |         return None
 252 | 
```

| Line | Explanation |
| --- | --- |
| 236 | Fold literal integer expressions and remove safe integer identities, leaving floating operations unchanged. |
| 237 | Extract optional literal values without parsing PTX strings. |
| 238 | Reject a statically known zero integer divisor. |
| 239 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 240 | Fold only when both inputs are exact integer literals. |
| 241 | Use the supported literal-folding operator table. |
| 242 | Evaluate the literal integer operation using floor semantics where applicable. |
| 243 | Normalize integer arithmetic to the signed 32-bit representation. |
| 244 | Return the folded integer as an immediate instead of a virtual register. |
| 245 | Remove a literal left zero for addition or a literal left one for multiplication. |
| 246 | Return the completed result to the caller of this function. |
| 247 | Remove right-side zero additions/subtractions, multiplication by one and division by one. |
| 248 | Return the completed result to the caller of this function. |
| 249 | Replace integer multiplication by zero or modulo one with an exact zero. |
| 250 | Return the completed result to the caller of this function. |
| 251 | Return the completed result to the caller of this function. |

#### integer_divmod at lines 253 to 270

Implement Emmy floor division and modulo while using short PTX forms where signs are proven safe.

```python
 253 |     def integer_divmod(self, op, left, right):
 254 |         # PTX truncates toward zero; Emmy integer /, // and % use floor semantics.
 255 |         if left.minimum is not None and left.minimum >= 0 and right.minimum is not None and right.minimum > 0:
 256 |             code = "rem" if op == "%" else "div"
 257 |             upper = right.maximum-1 if op == "%" else left.maximum // right.minimum
 258 |             return self.instruction("s32", f"{code}.s32", left, right, bounds=(0, upper))
 259 |         remainder = self.instruction("s32", "rem.s32", left, right)
 260 |         nonzero = self.instruction("pred", "setp.ne.s32", remainder, 0)
 261 |         signs = self.instruction("s32", "xor.b32", left, right)
 262 |         negative = self.instruction("pred", "setp.lt.s32", signs, 0)
 263 |         adjust = self.instruction("pred", "and.pred", nonzero, negative)
 264 |         if op == "%":
 265 |             correction = self.instruction("s32", "selp.s32", right, 0, adjust)
 266 |             return self.instruction("s32", "add.s32", remainder, correction)
 267 |         quotient = self.instruction("s32", "div.s32", left, right)
 268 |         correction = self.instruction("s32", "selp.s32", 1, 0, adjust)
 269 |         return self.instruction("s32", "sub.s32", quotient, correction)
 270 | 
```

| Line | Explanation |
| --- | --- |
| 253 | Implement Emmy floor division and modulo while using short PTX forms where signs are proven safe. |
| 254 | Document the adjacent implementation rule or invocation. |
| 255 | Skip the signed-floor correction only when both intervals prove a nonnegative numerator and positive divisor. |
| 256 | Choose the short remainder or quotient form when truncation and floor are proven equivalent. |
| 257 | Retain a conservative upper bound for the resulting nonnegative quotient or remainder. |
| 258 | Emit the proven-safe short path with a nonnegative result interval. |
| 259 | Compute PTX remainder, whose sign follows the numerator before correction. |
| 260 | Avoid correcting an exact division with zero remainder. |
| 261 | The sign bit of XOR identifies differing numerator/divisor signs. |
| 262 | Turn the differing-sign test into a predicate. |
| 263 | Correct only nonexact divisions with opposite signs. |
| 264 | Check the stated condition before executing the following guarded operation. |
| 265 | Choose one divisor as the remainder correction, or zero. |
| 266 | Produce divisor-signed modulo consistent with floor division. |
| 267 | Compute the initial quotient truncated toward zero. |
| 268 | Choose whether the quotient needs a one-unit decrement. |
| 269 | Convert truncation to floor for the signed quotient. |

#### compute at lines 271 to 297

```python
 271 |     def compute(self, op: str, args: list[Value]) -> Value:
 272 |         if any(a.dtype != "f32" for a in args):
 273 |             raise CompileError(f"{op}: only f32 arithmetic is supported")
 274 |         if op in ELEMENTWISE_BINARY:
 275 |             return self.binary(ELEMENTWISE_BINARY[op], *args)
 276 |         if op == "copy":
 277 |             return args[0]
 278 |         if op in ("maximum", "minimum"):
 279 |             return self.instruction("f32", "max.f32" if op == "maximum" else "min.f32", *args)
 280 |         if op == "relu":
 281 |             return self.instruction("f32", "max.f32", args[0], self.constant(0))
 282 |         if op == "reciprocal":
 283 |             return self.binary("/", self.constant(1), args[0])
 284 |         if op == "sqrt":
 285 |             return self.instruction("f32", "sqrt.rn.f32", *args)
 286 |         if op == "rsqrt":
 287 |             return self.instruction("f32", "rsqrt.approx.f32", *args)
 288 |         if op in ("exp", "log"):
 289 |             if op == "exp":
 290 |                 scaled = self.binary("*", args[0], self.constant(math.log2(math.e)))
 291 |                 return self.instruction("f32", "ex2.approx.f32", scaled)
 292 |             value = self.instruction("f32", "lg2.approx.f32", args[0])
 293 |             return self.binary("*", value, self.constant(math.log(2)))
 294 |         if op in ("negative", "neg", "abs"):
 295 |             return self.instruction("f32", "abs.f32" if op == "abs" else "neg.f32", *args)
 296 |         raise CompileError(f"unsupported elementwise operation {op!r}")
 297 | 
```

| Line | Explanation |
| --- | --- |
| 271 | Define `compute` with the parameters and result contract shown in this signature. |
| 272 | Elementwise data arithmetic is implemented only for float32 operands. |
| 273 | Reject another arithmetic dtype explicitly. |
| 274 | Reuse the shared scalar arithmetic spelling map. |
| 275 | Return the completed result to the caller of this function. |
| 276 | Copy is an identity operation before the named destination is bound. |
| 277 | Reuse the source operand and let bind perform any needed move. |
| 278 | Recognize the two extrema operations used by reductions and pointwise code. |
| 279 | Select a float32 min or max instruction. |
| 280 | ReLU is implemented as a maximum against zero. |
| 281 | Emit the zero clamp at float32 precision. |
| 282 | Reciprocal is expressed as an ordinary division. |
| 283 | Use 1 divided by the input with the binary division's rounding behavior. |
| 284 | Recognize the ordinary square-root operation. |
| 285 | Emit nearest-rounding float32 square root. |
| 286 | Recognize reciprocal square root as a separate operation. |
| 287 | Emit the hardware approximate reciprocal-square-root instruction. |
| 288 | Recognize natural exponential and logarithm, which need base conversion. |
| 289 | Separate the exponential's input scale from the logarithm's output scale. |
| 290 | Multiply the natural-exponential argument by log2(e). |
| 291 | Apply base-two exponential to obtain the approximation to exp(x). |
| 292 | Compute a base-two logarithm of the input. |
| 293 | Multiply log2(x) by ln(2) to obtain the natural logarithm. |
| 294 | Accept the registered negation spellings and absolute value. |
| 295 | Select a float32 sign or absolute-value instruction. |
| 296 | Fail on an unimplemented elementwise operation without using a fallback renderer. |

#### prepare_shared at lines 298 to 331

```python
 298 |     def prepare_shared(self):
 299 |         offset = 0
 300 |         specs = {}
 301 |         for stmt in walk(self.kernel.body):
 302 |             if not isinstance(stmt, Smem):
 303 |                 continue
 304 |             symbol(stmt.name)
 305 |             if stmt.dtype not in ("float", "f32"):
 306 |                 raise CompileError(f"shared buffer {stmt.name}: unsupported type {stmt.dtype}")
 307 |             alignment = max(stmt.align, 4)
 308 |             if alignment & (alignment - 1):
 309 |                 raise CompileError("shared memory alignment must be a power of two")
 310 |             offset = (offset + alignment - 1) // alignment * alignment
 311 |             if stmt.name in specs:
 312 |                 raise CompileError(f"duplicate shared buffer {stmt.name}")
 313 |             specs[stmt.name] = (stmt, offset)
 314 |             self.buffers[stmt.name] = (stmt.extents, "f32")
 315 |             offset += math.prod(stmt.extents) * 4
 316 |         dynamic = offset > 48 * 1024
 317 |         if dynamic and self.launch["smem_bytes"] < offset:
 318 |             raise CompileError(f"launch needs at least {offset} bytes of dynamic shared memory")
 319 |         if dynamic:
 320 |             alignment = max(16, *(max(stmt.align, 4) for stmt, _ in specs.values()))
 321 |             self.module_declarations.append(f".extern .shared .align {alignment} .b8 shared_pool[];")
 322 |         for name, (stmt, start) in specs.items():
 323 |             base = self.reg("s32")
 324 |             self.shared[name] = base
 325 |             if dynamic:
 326 |                 self.emit(f"mov.u32 {base.text}, shared_pool;")
 327 |                 self.emit(f"add.u32 {base.text}, {base.text}, {start};")
 328 |             else:
 329 |                 self.declarations.append(f".shared .align {max(stmt.align, 4)} .b8 smem_{name}[{math.prod(stmt.extents) * 4}];")
 330 |                 self.emit(f"mov.u32 {base.text}, smem_{name};")
 331 | 
```

| Line | Explanation |
| --- | --- |
| 298 | Define `prepare_shared` with the parameters and result contract shown in this signature. |
| 299 | Start shared-memory packing at byte offset zero. |
| 300 | Record each declaration and its aligned starting offset. |
| 301 | Find shared declarations anywhere in the structured body. |
| 302 | Skip statements that do not declare a shared buffer. |
| 303 | Continue collecting without emitting body instructions for unrelated nodes. |
| 304 | Ensure each shared symbol can be embedded safely in a PTX declaration. |
| 305 | Accept the IR's two spellings for float32 shared storage. |
| 306 | Reject an unsupported shared element type. |
| 307 | Require at least the natural four-byte alignment of float32. |
| 308 | Check the bit property that characterizes positive powers of two. |
| 309 | Reject an alignment that cannot be used in this packing scheme. |
| 310 | Round the next free byte offset upward to the requested alignment. |
| 311 | Detect redeclaration of a shared-memory name in the same kernel. |
| 312 | Reject overlapping logical declarations rather than silently reusing storage. |
| 313 | Remember the typed declaration and its slice offset. |
| 314 | Register the shared buffer's shape for the common address flattener. |
| 315 | Advance the allocation end by element count times four bytes. |
| 316 | Match the repository's threshold for switching to a dynamic shared pool. |
| 317 | Check that the prescribed launch provides the required dynamic storage. |
| 318 | Report the required byte count if the launch is too small. |
| 319 | Use one external pool only when the static allocation limit is exceeded. |
| 320 | Choose a base alignment sufficient for every packed shared slice. |
| 321 | Declare dynamic shared storage at module scope, outside the entry function. |
| 322 | Create an address for every logical shared buffer. |
| 323 | Allocate a 32-bit shared-space pointer register. |
| 324 | Associate that register with the buffer's logical name. |
| 325 | Select dynamic-slice address construction for large shared allocations. |
| 326 | Load the dynamic pool's shared-space base address. |
| 327 | Add this buffer's aligned byte offset within the pool. |
| 328 | Enter the alternative path when the preceding condition does not hold. |
| 329 | Declare a naturally or explicitly aligned static byte array of the required size. |
| 330 | Load the static array's shared-space address. |

#### address at lines 332 to 344

```python
 332 |     def address(self, buffer: str, index) -> Value:
 333 |         shape, _ = self.buffers[buffer]
 334 |         if len(index) != len(shape) and not (len(index) == 0 and math.prod(shape) == 1):
 335 |             raise CompileError(f"{buffer}: index rank {len(index)} differs from shape {shape}")
 336 |         flat = self.constant(0, "s32")
 337 |         for i, coordinate in enumerate(index):
 338 |             flat = self.binary("+", self.binary("*", flat, self.constant(shape[i], "s32")), self.expr(coordinate))
 339 |         if buffer in self.shared:
 340 |             offset = self.instruction("s32", "mul.lo.u32", flat, 4)
 341 |             return self.instruction("s32", "add.u32", self.shared[buffer], offset)
 342 |         offset = self.instruction("u64", "mul.wide.s32", flat, 4)
 343 |         return self.instruction("u64", "add.u64", self.pointers[buffer], offset)
 344 | 
```

| Line | Explanation |
| --- | --- |
| 332 | Define `address` with the parameters and result contract shown in this signature. |
| 333 | Retrieve the tensor dimensions used to flatten the coordinates. |
| 334 | Require the correct index rank, allowing an empty index for a one-element scalar output. |
| 335 | Explain an invalid coordinate rank before generating an unsafe memory access. |
| 336 | Initialize the row-major element offset to zero. |
| 337 | Process coordinates from outermost to innermost dimension. |
| 338 | Apply Horner-form flattening: multiply the prefix by the dimension, then add its coordinate. |
| 339 | Distinguish 32-bit shared addresses from 64-bit global addresses. |
| 340 | Convert the shared element offset to a four-byte-scaled byte offset. |
| 341 | Add that offset to the shared buffer's base. |
| 342 | Widen the signed element offset while multiplying by four for global addressing. |
| 343 | Add the byte offset to the pointer in the separate buffer-pointer map. |

#### load at lines 345 to 356

```python
 345 |     def load(self, buffer, index, names):
 346 |         address = self.address(buffer, index)
 347 |         space = "shared" if buffer in self.shared else "global"
 348 |         if len(names) not in (1, 2, 4):
 349 |             raise CompileError("loads must contain 1, 2 or 4 values")
 350 |         regs = [self.reg("f32") for _ in names]
 351 |         suffix = "" if len(names) == 1 else f".v{len(names)}"
 352 |         dest = regs[0].text if len(names) == 1 else "{" + ", ".join(r.text for r in regs) + "}"
 353 |         self.emit(f"ld.{space}{suffix}.f32 {dest}, [{address.text}];")
 354 |         for name, reg in zip(names, regs):
 355 |             self.bind(name, reg)
 356 | 
```

| Line | Explanation |
| --- | --- |
| 345 | Define `load` with the parameters and result contract shown in this signature. |
| 346 | Calculate the first element's byte address using the common flattener. |
| 347 | Choose ld.shared or ld.global from the buffer's storage class. |
| 348 | Restrict grouped loads to supported scalar, pair and four-value widths. |
| 349 | Reject an unsupported vector width explicitly. |
| 350 | Reserve one float register for each loaded element. |
| 351 | Choose no vector suffix for scalar access, otherwise v2 or v4. |
| 352 | Format scalar destinations or the PTX register tuple required by vector loads. |
| 353 | Emit the memory instruction at the computed address. |
| 354 | Associate each returned element with its corresponding IR name. |
| 355 | Bind the loaded value into a stable named register. |

#### store at lines 357 to 373

```python
 357 |     def store(self, buffer, index, values, atomic=False):
 358 |         if any(self.env[name].dtype != "f32" for name in values):
 359 |             raise CompileError("float32 stores require float32 values")
 360 |         address = self.address(buffer, index)
 361 |         space = "shared" if buffer in self.shared else "global"
 362 |         if atomic:
 363 |             if space != "global":
 364 |                 raise CompileError("only global atomic adds are supported")
 365 |             for i, name in enumerate(values):
 366 |                 self.emit(f"red.global.add.f32 [{address.text}+{i * 4}], {self.env[name].text};")
 367 |             return
 368 |         if len(values) not in (1, 2, 4):
 369 |             raise CompileError("stores must contain 1, 2 or 4 values")
 370 |         suffix = "" if len(values) == 1 else f".v{len(values)}"
 371 |         src = self.env[values[0]].text if len(values) == 1 else "{" + ", ".join(self.env[v].text for v in values) + "}"
 372 |         self.emit(f"st.{space}{suffix}.f32 [{address.text}], {src};")
 373 | 
```

| Line | Explanation |
| --- | --- |
| 357 | Define `store` with the parameters and result contract shown in this signature. |
| 358 | Reject raw integer-bit storage through a float32 store instruction. |
| 359 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 360 | Compute the first output element's byte address. |
| 361 | Select shared or global storage independently of whether this is a vector write. |
| 362 | Separate atomic accumulation from ordinary memory stores. |
| 363 | Only global atomic addition is implemented in this backend. |
| 364 | Reject an unsupported shared atomic update. |
| 365 | Handle each atomic vector component as a separate scalar atomic add. |
| 366 | Emit an atomic addition without a returned old value, advancing by four bytes per component. |
| 367 | Finish the atomic path without also issuing a normal store. |
| 368 | Restrict ordinary stores to the supported vector widths. |
| 369 | Reject an unsupported store width. |
| 370 | Choose the scalar, v2 or v4 store instruction form. |
| 371 | Format one source register or a tuple in the IR's value order. |
| 372 | Emit the store to the selected address space. |

#### barrier at lines 374 to 381

```python
 374 |     def barrier(self, barrier_id=0, count=None, warp=False):
 375 |         if warp:
 376 |             mask = self.instruction("s32", "activemask.b32")
 377 |             self.emit(f"bar.warp.sync {mask.text};")
 378 |         else:
 379 |             tail = "" if count is None else f", {count}"
 380 |             self.emit(f"bar.sync {barrier_id}{tail};")
 381 | 
```

| Line | Explanation |
| --- | --- |
| 374 | Define `barrier` with the parameters and result contract shown in this signature. |
| 375 | Distinguish warp-scope synchronization from CTA/named barriers. |
| 376 | Capture the mask of currently participating warp lanes. |
| 377 | Synchronize those warp lanes with a warp barrier. |
| 378 | Enter the alternative path when the preceding condition does not hold. |
| 379 | Include an explicit participating-thread count only when supplied by the IR. |
| 380 | Emit the default CTA or named barrier with the requested ID and count. |

#### loop_bindings at lines 382 to 397

Identify lexical locals and reduction exports without descending into a nested loop.

```python
 382 |     def loop_bindings(self, body):
 383 |         local, accumulators = set(), {}
 384 |         for stmt in body:
 385 |             if isinstance(stmt, Accum):
 386 |                 accumulators[stmt.name] = stmt
 387 |             elif isinstance(stmt, Load):
 388 |                 local.update(stmt.names)
 389 |             elif isinstance(stmt, (Assign, Init, Let)):
 390 |                 local.add(stmt.name)
 391 |             elif isinstance(stmt, Cond):
 392 |                 for branch in (stmt.body, stmt.else_body):
 393 |                     names, states = self.loop_bindings(branch)
 394 |                     local.update(names)
 395 |                     accumulators.update(states)
 396 |         return local - accumulators.keys(), accumulators
 397 | 
```

| Line | Explanation |
| --- | --- |
| 382 | Identify lexical locals and reduction exports without descending into a nested loop. |
| 383 | Track ordinary loop-local definitions separately from carried reduction exports. |
| 384 | Scan the current loop's statements before emitting the loop header. |
| 385 | Check the stated condition before executing the following guarded operation. |
| 386 | Remember the reduction update that determines a carried name and its identity. |
| 387 | Select the next supported case when the preceding cases did not match. |
| 388 | Record every name introduced by a scalar or vector load in this loop scope. |
| 389 | Recognize ordinary local definitions that must shadow outer names. |
| 390 | Mark this definition as local to the loop. |
| 391 | Accumulators under conditions still belong to the surrounding loop. |
| 392 | Inspect condition branches without crossing nested loop boundaries. |
| 393 | Collect definitions and carried state from that conditional body. |
| 394 | Merge the conditional body's local names into this loop scope. |
| 395 | Merge the conditional body's carried reductions into this loop's exports. |
| 396 | Separate local names from accumulator names that remain visible after the loop. |

#### loop at lines 398 to 434

```python
 398 |     def loop(self, stmt):
 399 |         strided = isinstance(stmt, StridedLoop)
 400 |         start = self.expr(stmt.start) if strided else self.constant(0, "s32")
 401 |         step = self.expr(stmt.step) if strided else self.constant(1, "s32")
 402 |         end = self.expr(stmt.end) if strided and stmt.end is not None else self.constant(stmt.axis.extent.as_static(), "s32")
 403 |         if start.dtype != "s32" or step.dtype != "s32" or end.dtype != "s32":
 404 |             raise CompileError("loop bounds must be integers")
 405 |         if step.integer is not None and step.integer <= 0:
 406 |             raise CompileError("loop stride must be positive")
 407 |         outer = self.env
 408 |         self.env = outer.copy()
 409 |         local, accumulators = self.loop_bindings(stmt.body)
 410 |         for name in local | {stmt.axis.name}:
 411 |             self.env.pop(name, None)
 412 |         if stmt.seed:
 413 |             for name, accumulator in accumulators.items():
 414 |                 if accumulator.op.identity is None:
 415 |                     raise CompileError(f"{accumulator.op.name} has no reduction identity")
 416 |                 self.env.pop(name, None)
 417 |                 self.bind(name, self.constant(accumulator.op.identity))
 418 |         self.bind(stmt.axis.name, start)
 419 |         induction = self.env[stmt.axis.name]
 420 |         if start.minimum is not None and end.maximum is not None and start.minimum < end.maximum:
 421 |             induction = Value(induction.text, induction.dtype, None, start.minimum, end.maximum-1)
 422 |             self.env[stmt.axis.name] = induction
 423 |         top, done = self.label("loop"), self.label("done")
 424 |         self.emit(top + ":")
 425 |         stop = self.binary(">=", induction, end)
 426 |         self.emit(f"@{stop.text} bra {done};")
 427 |         self.body(stmt.body)
 428 |         self.emit(f"add.s32 {induction.text}, {induction.text}, {step.text};")
 429 |         self.emit(f"bra {top};")
 430 |         self.emit(done + ":")
 431 |         exports = {name: self.env[name] for name in accumulators}
 432 |         self.env = outer
 433 |         self.env.update(exports)
 434 | 
```

| Line | Explanation |
| --- | --- |
| 398 | Define `loop` with the parameters and result contract shown in this signature. |
| 399 | Distinguish explicit-stride loops from zero-based unit-step loops. |
| 400 | Use the thread-dependent start for strided loops or zero for ordinary loops. |
| 401 | Use the IR stride or a unit increment. |
| 402 | Use an explicit strided end when supplied, otherwise the static axis extent. |
| 403 | Check the stated condition before executing the following guarded operation. |
| 404 | Reject bounds that would select invalid integer loop instructions. |
| 405 | Reject a nonpositive literal stride rather than generating a nonadvancing loop. |
| 406 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 407 | Retain the complete enclosing lexical environment. |
| 408 | Create a loop-specific value mapping while preserving readable outer bindings. |
| 409 | Identify which names must be fresh locals and which reductions are exported. |
| 410 | Include the induction variable among names that shadow an enclosing binding. |
| 411 | Remove the copied binding so the new local gets a distinct register. |
| 412 | Check the stated condition before executing the following guarded operation. |
| 413 | Initialize the reductions owned by a seeded loop. |
| 414 | Require an operation identity before seeding carried state. |
| 415 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 416 | Remove the copied binding so the new local gets a distinct register. |
| 417 | Initialize the loop's own reduction register to its exact neutral element. |
| 418 | Allocate and initialize this loop's own induction register. |
| 419 | Hold the induction register while the body may create other bindings. |
| 420 | Retain a safe induction interval when the start and exclusive end are known. |
| 421 | Tag the induction register with the supported positive-stride loop interval. |
| 422 | Make those loop-coordinate bounds available to body address arithmetic. |
| 423 | Allocate unique loop-header and exit labels. |
| 424 | Place the header label before the bound comparison, so each iteration retests it. |
| 425 | Compare the current induction value with the exclusive upper bound. |
| 426 | Exit before executing the body when the limit is reached. |
| 427 | Lower the loop body once; runtime control flow repeats those instructions. |
| 428 | Advance the induction register by the actual loop stride. |
| 429 | Branch back to the bound comparison. |
| 430 | Place the exit label after the back edge. |
| 431 | Capture only this loop's carried output bindings. |
| 432 | Restore all enclosing locals, including values shadowed inside the loop. |
| 433 | Publish completed reductions without leaking ordinary local definitions. |

#### conditional at lines 435 to 443

```python
 435 |     def conditional(self, pred: Value, body, else_body=()):
 436 |         other, done = self.label("else"), self.label("endif")
 437 |         self.emit(f"@!{pred.text} bra {other};")
 438 |         self.body(body)
 439 |         self.emit(f"bra {done};")
 440 |         self.emit(other + ":")
 441 |         self.body(else_body)
 442 |         self.emit(done + ":")
 443 | 
```

| Line | Explanation |
| --- | --- |
| 435 | Define `conditional` with the parameters and result contract shown in this signature. |
| 436 | Create separate false-branch and join labels. |
| 437 | Skip the true body when the predicate is false. |
| 438 | Emit the true branch in source order. |
| 439 | Skip the false body after completing the true branch. |
| 440 | Mark where the false branch begins. |
| 441 | Emit the false branch, which may be empty. |
| 442 | Mark the point at which both paths rejoin. |

#### tile at lines 444 to 466

```python
 444 |     def tile(self, stmt: Tile):
 445 |         # Raster hints only permute independent CTAs. Row-major visits the same cells.
 446 |         block = stmt.block_threads
 447 |         if stmt.aux_threads and not block:
 448 |             raise CompileError("auxiliary threads require block_threads")
 449 |         local = self.instruction("s32", "rem.u32", self.tid, block) if stmt.aux_threads else self.tid
 450 |         gid = self.instruction("s32", "mad.lo.s32", self.bid, block or self.ntid, local)
 451 |         extents = [a.extent.as_static() for a in stmt.axes]
 452 |         total = math.prod(extents)
 453 |         if any(e <= 0 for e in extents) or total >= (1 << 31):
 454 |             raise CompileError("tile extents must be positive and fit signed 32-bit indexing")
 455 |         done = self.label("tile_end")
 456 |         out = self.binary(">=", gid, self.constant(total, "s32"))
 457 |         self.emit(f"@{out.text} bra {done};")
 458 |         stride = total
 459 |         for axis, extent in zip(stmt.axes, extents):
 460 |             stride //= extent
 461 |             coordinate = self.instruction("s32", "div.u32", gid, stride)
 462 |             coordinate = self.instruction("s32", "rem.u32", coordinate, extent, bounds=(0, extent-1))
 463 |             self.bind(axis.name, coordinate)
 464 |         self.body(stmt.body)
 465 |         self.emit(done + ":")
 466 | 
```

| Line | Explanation |
| --- | --- |
| 444 | Define `tile` with the parameters and result contract shown in this signature. |
| 445 | Document the intentional replacement of a cache-locality permutation with equivalent row-major coverage. |
| 446 | Read the cooperative compute-band width, if the tile has one. |
| 447 | Detect an auxiliary producer band without a compute-band size. |
| 448 | Reject the undefined coordinate mapping for that combination. |
| 449 | Wrap auxiliary thread coordinates onto compute cells when producer threads are present. |
| 450 | Form the global work index from block index, compute-band width and local coordinate. |
| 451 | Read the static iteration-space dimensions from typed axes. |
| 452 | Calculate total useful thread cells, not the number of original tensor elements. |
| 453 | Require positive axis extents and a tile size representable by signed 32-bit indexing. |
| 454 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 455 | Create the label used to skip unused launch threads. |
| 456 | Detect work indices outside the tile iteration space. |
| 457 | Branch around the body for excess threads before they can load or store memory. |
| 458 | Start the suffix-product calculation used for coordinate decoding. |
| 459 | Decode the tile's axes in their declared order. |
| 460 | Obtain the product of extents strictly inside this axis. |
| 461 | Divide the global index by the inner stride. |
| 462 | Isolate the axis coordinate and retain its exact nonnegative interval. |
| 463 | Bind the coordinate under the axis's IR name. |
| 464 | Lower the scheduled per-cell work using those coordinates. |
| 465 | Rejoin threads that skipped unused cells. |

#### shuffle at lines 467 to 480

```python
 467 |     def shuffle(self, stmt: WarpShuffle):
 468 |         if not stmt.state or len(stmt.state) != len(stmt.state_b):
 469 |             raise CompileError("shuffle state component counts must match")
 470 |         if stmt.dtype.name != "f32" or stmt.length < 1 or stmt.length > 32 or stmt.length & (stmt.length - 1):
 471 |             raise CompileError("warp shuffle needs f32 and a power of two length up to 32")
 472 |         mask = self.instruction("s32", "activemask.b32")
 473 |         distance = stmt.length // 2
 474 |         while distance:
 475 |             for name, other in zip(stmt.state, stmt.state_b):
 476 |                 value = self.instruction("f32", "shfl.sync.bfly.b32", self.env[name], distance, 31, mask)
 477 |                 self.bind(other, value)
 478 |             self.body(stmt.combine_states)
 479 |             distance //= 2
 480 | 
```

| Line | Explanation |
| --- | --- |
| 467 | Define `shuffle` with the parameters and result contract shown in this signature. |
| 468 | Reject missing or unequal shuffle state components before zip can truncate them. |
| 469 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 470 | Restrict warp butterflies to supported float32 power-of-two widths within one warp. |
| 471 | Reject an unsupported shuffle configuration. |
| 472 | Capture the participating-lane mask before the shuffle ladder. |
| 473 | Start with the largest XOR distance in the butterfly. |
| 474 | Generate one stage for each decreasing power-of-two distance. |
| 475 | Snapshot every component of the partner lane's state before running the combine program. |
| 476 | Read the XOR partner's register with the same active mask. |
| 477 | Bind the partner's value under the IR's state_b name. |
| 478 | Execute the supplied combine assignments, including any coupled softmax state. |
| 479 | Halve the distance until the whole requested subgroup has been combined. |

#### tree at lines 481 to 511

```python
 481 |     def tree(self, stmt: TreeHalve):
 482 |         if not stmt.state or not len(stmt.bufs) == len(stmt.state) == len(stmt.state_b):
 483 |             raise CompileError("shared tree state component counts must match")
 484 |         if stmt.length < 1 or stmt.length & (stmt.length - 1):
 485 |             raise CompileError("shared reduction length must be a power of two")
 486 |         tid = self.env[stmt.tid_var]
 487 |         scale = stmt.inner[1] if stmt.inner else 1
 488 |         inner = Var(stmt.inner[0]) if stmt.inner else Literal(0, "int")
 489 |         def index(offset):
 490 |             return (BinaryExpr("+", BinaryExpr("*", offset, Literal(scale, "int")), inner),)
 491 |         distance = stmt.length // 2
 492 |         while distance:
 493 |             skip = self.label("tree_skip")
 494 |             active = self.binary("<", tid, self.constant(distance, "s32"))
 495 |             if stmt.tid_var == "warp":
 496 |                 leader = self.binary("==", self.env["lane"], self.constant(0, "s32"))
 497 |                 active = self.binary("&&", active, leader)
 498 |             self.emit(f"@!{active.text} bra {skip};")
 499 |             for buf, name, other in zip(stmt.bufs, stmt.state, stmt.state_b):
 500 |                 self.load(buf, index(Var(stmt.tid_var)), (name,))
 501 |                 self.load(buf, index(BinaryExpr("+", Var(stmt.tid_var), Literal(distance, "int"))), (other,))
 502 |             self.body(stmt.combine_states)
 503 |             for buf, name in zip(stmt.bufs, stmt.state):
 504 |                 self.store(buf, index(Var(stmt.tid_var)), (name,))
 505 |             self.emit(skip + ":")
 506 |             self.barrier(stmt.barrier_id, stmt.barrier_count)
 507 |             distance //= 2
 508 |         for buf, name in zip(stmt.bufs, stmt.state):
 509 |             self.load(buf, (inner,), (name,))
 510 |         self.barrier(stmt.barrier_id, stmt.barrier_count)
 511 | 
```

| Line | Explanation |
| --- | --- |
| 481 | Define `tree` with the parameters and result contract shown in this signature. |
| 482 | Require matching, nonempty shared buffers, local states and partner states. |
| 483 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 484 | Check the shared tree's required power-of-two length. |
| 485 | Reject a tree shape that this halving algorithm cannot reduce. |
| 486 | Use the IR-selected participant index, which may identify a thread or a warp. |
| 487 | Use the segment width for a segmented tree, otherwise one slot per participant. |
| 488 | Select the inner slot expression or the scalar slot zero. |
| 489 | Define `index` with the parameters and result contract shown in this signature. |
| 490 | Build the flat shared-slot expression participant times segment width plus inner slot. |
| 491 | Start by combining the lower and upper halves of the populated tree. |
| 492 | Generate shared-tree stages until only the root remains. |
| 493 | Create a label for participants that do not update at this level. |
| 494 | Select the participants in the lower half of this stage. |
| 495 | Warp-indexed trees require only one writer per warp slot. |
| 496 | Select lane zero within each participating warp. |
| 497 | Combine the lower-half condition with the single-writer condition. |
| 498 | Have all other threads bypass memory updates while still reaching the barrier. |
| 499 | Load both states for every component before applying the merge program. |
| 500 | Load the lower-half shared partial into its state register. |
| 501 | Load the matching upper-half partial into the partner register. |
| 502 | Lower the exact supplied state-combination program. |
| 503 | Write every merged component back to its lower-half slot. |
| 504 | Store the reduced state in shared memory for the next tree level. |
| 505 | Join inactive participants immediately before synchronization. |
| 506 | Make the current level's writes visible to all required participants. |
| 507 | Reduce the number of active tree slots by half. |
| 508 | Broadcast the completed root state to every participating thread. |
| 509 | Load slot zero, or the requested segmented inner slot, into each state component. |
| 510 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |

#### typed_value at lines 512 to 524

Honor a declared scalar dtype through the supported numeric conversions.

```python
 512 |     def typed_value(self, value, dtype):
 513 |         target = getattr(dtype, "name", dtype)
 514 |         target = {"i32": "s32", "int": "s32"}.get(target, target)
 515 |         if target is None or target == value.dtype:
 516 |             return value
 517 |         if target == "f32" and value.dtype == "s32":
 518 |             if value.integer is not None:
 519 |                 return self.constant(value.integer)
 520 |             return self.instruction("f32", "cvt.rn.f32.s32", value)
 521 |         if target == "s32" and value.dtype == "f32":
 522 |             return self.instruction("s32", "cvt.rzi.s32.f32", value)
 523 |         raise CompileError(f"unsupported value conversion {value.dtype} to {target}")
 524 | 
```

| Line | Explanation |
| --- | --- |
| 512 | Honor a declared scalar dtype through the supported numeric conversions. |
| 513 | Normalize the requested declared scalar type. |
| 514 | Use the backend's signed 32-bit spelling for integer declarations. |
| 515 | Leave unspecified or already matching scalar types unchanged. |
| 516 | Return the completed result to the caller of this function. |
| 517 | Convert an integer expression into a declared float32 value. |
| 518 | Encode a known integer directly as a float literal without a conversion instruction. |
| 519 | Preserve numeric value while changing the literal representation to float32. |
| 520 | Convert a computed integer register to float32 with nearest rounding. |
| 521 | Support a declared integer binding from a computed float expression. |
| 522 | Use truncating float-to-integer conversion for the supported scalar cast. |
| 523 | Reject this unsupported or malformed input with the specific diagnostic shown. |

#### body at lines 525 to 580

```python
 525 |     def body(self, body):
 526 |         for stmt in body:
 527 |             self.emit(f"// {type(stmt).__name__}")
 528 |             dtype = getattr(stmt, "dtype", getattr(stmt, "value_dtype", None))
 529 |             if not isinstance(stmt, (Let, Smem)) and dtype is not None and dtype.name != "f32":
 530 |                 raise CompileError(f"{type(stmt).__name__}: only f32 data is supported")
 531 |             if isinstance(stmt, Tile):
 532 |                 self.tile(stmt)
 533 |             elif isinstance(stmt, (Loop, StridedLoop)):
 534 |                 self.loop(stmt)
 535 |             elif isinstance(stmt, Load):
 536 |                 self.load(stmt.input, stmt.index, stmt.names)
 537 |             elif isinstance(stmt, Write):
 538 |                 if stmt.swizzle != "NONE":
 539 |                     raise CompileError("swizzled stores are not supported")
 540 |                 self.store(stmt.output, stmt.index, stmt.values, stmt.atomic)
 541 |             elif isinstance(stmt, Assign):
 542 |                 self.bind(stmt.name, self.compute(stmt.op.name, [self.env[a] for a in stmt.args]))
 543 |             elif isinstance(stmt, Accum):
 544 |                 base = stmt.base or stmt.name
 545 |                 self.bind(stmt.name, self.compute(stmt.op.name, [self.env[base], self.env[stmt.value]]))
 546 |             elif isinstance(stmt, Init):
 547 |                 self.bind(stmt.name, self.constant(stmt.identity))
 548 |             elif isinstance(stmt, Let):
 549 |                 self.bind(stmt.name, self.typed_value(self.expr(stmt.value), stmt.dtype))
 550 |             elif isinstance(stmt, Cond):
 551 |                 self.conditional(self.predicate(self.expr(stmt.cond)), stmt.body, stmt.else_body)
 552 |             elif isinstance(stmt, Smem):
 553 |                 pass  # All shared declarations and base pointers are prepared in the prologue.
 554 |             elif isinstance(stmt, Sync):
 555 |                 self.barrier(stmt.barrier_id, stmt.count, stmt.warp)
 556 |             elif isinstance(stmt, WarpShuffle):
 557 |                 self.shuffle(stmt)
 558 |             elif isinstance(stmt, TreeHalve):
 559 |                 self.tree(stmt)
 560 |             elif isinstance(stmt, CpAsyncCopy):
 561 |                 if stmt.swizzle != "NONE":
 562 |                     raise CompileError("swizzled async copies are not supported")
 563 |                 if stmt.nbytes not in (4, 8, 16):
 564 |                     raise CompileError("async copies must contain 4, 8 or 16 bytes")
 565 |                 index = stmt.smem_index
 566 |                 if stmt.lane_index is not None:
 567 |                     if len(index) != len(stmt.lane_index):
 568 |                         raise CompileError("async copy lane index rank differs from slab rank")
 569 |                     index = tuple(BinaryExpr("+", a, b) for a, b in zip(index, stmt.lane_index))
 570 |                 dst = self.address(stmt.smem, index)
 571 |                 src = self.address(stmt.src, stmt.src_index)
 572 |                 cache = "cg" if stmt.nbytes == 16 else "ca"
 573 |                 self.emit(f"cp.async.{cache}.shared.global [{dst.text}], [{src.text}], {stmt.nbytes};")
 574 |             elif isinstance(stmt, CpAsyncCommit):
 575 |                 self.emit("cp.async.commit_group;")
 576 |             elif isinstance(stmt, CpAsyncWait):
 577 |                 self.emit(f"cp.async.wait_group {stmt.group};")
 578 |             else:
 579 |                 raise CompileError(f"unsupported statement {type(stmt).__name__}")
 580 | 
```

| Line | Explanation |
| --- | --- |
| 525 | Define `body` with the parameters and result contract shown in this signature. |
| 526 | Lower statements sequentially so updates and synchronization retain their original ordering. |
| 527 | Include an ASCII comment identifying the source statement kind in the emitted PTX. |
| 528 | Inspect explicit statement element types before lowering its operation. |
| 529 | Reject unsupported data dtypes while allowing Let's supported scalar conversions and Smem's separate check. |
| 530 | Reject this unsupported or malformed input with the specific diagnostic shown. |
| 531 | Dispatch this typed statement to the corresponding lowering path below. |
| 532 | Delegate tile coordinate mapping and body lowering to the tile handler. |
| 533 | Dispatch this typed statement to the corresponding lowering path below. |
| 534 | Delegate both loop forms to the same seed-aware control-flow lowering. |
| 535 | Dispatch this typed statement to the corresponding lowering path below. |
| 536 | Pass the typed buffer name, coordinates and destination names to the memory handler. |
| 537 | Dispatch this typed statement to the corresponding lowering path below. |
| 538 | Detect a storage permutation that the ordinary address flattener does not implement. |
| 539 | Reject swizzled stores instead of writing incorrect shared slots. |
| 540 | Preserve the output coordinates, grouped values and atomic-add flag. |
| 541 | Dispatch this typed statement to the corresponding lowering path below. |
| 542 | Read each named argument, compute the elementwise operation and bind the destination. |
| 543 | Dispatch this typed statement to the corresponding lowering path below. |
| 544 | Use a rescaled base when the IR supplies one, otherwise the carried accumulator itself. |
| 545 | Apply the reduction operation to the selected base and incoming value, then update the carried name. |
| 546 | Dispatch this typed statement to the corresponding lowering path below. |
| 547 | Emit explicit state initialization where the schedule placed it. |
| 548 | Dispatch this typed statement to the corresponding lowering path below. |
| 549 | Honor Let's declared dtype rather than storing an integer literal as raw float bits. |
| 550 | Dispatch this typed statement to the corresponding lowering path below. |
| 551 | Lower both conditional bodies under a typed predicate. |
| 552 | Dispatch this typed statement to the corresponding lowering path below. |
| 553 | Do nothing here because shared storage declarations were already prepared before body lowering. |
| 554 | Dispatch this typed statement to the corresponding lowering path below. |
| 555 | Preserve barrier ID, participating count and warp scope from the IR. |
| 556 | Dispatch this typed statement to the corresponding lowering path below. |
| 557 | Delegate a cross-lane combine to the generic butterfly handler. |
| 558 | Dispatch this typed statement to the corresponding lowering path below. |
| 559 | Delegate a cross-warp/shared combine to the generic halving handler. |
| 560 | Dispatch this typed statement to the corresponding lowering path below. |
| 561 | Detect software-swizzled async destination layout. |
| 562 | Reject that unsupported address permutation. |
| 563 | Constrain async copies to PTX's supported byte sizes. |
| 564 | Report an invalid copy width before asking ptxas to assemble it. |
| 565 | Start with the destination slab's base coordinates. |
| 566 | Check whether this copy separately encodes its lane's within-slab coordinates. |
| 567 | Require the base and lane coordinate tuples to have the same rank. |
| 568 | Reject a malformed lane/slab combination rather than truncating zip silently. |
| 569 | Add each lane coordinate to the matching slab coordinate for an unswizzled copy. |
| 570 | Calculate the resulting 32-bit destination shared address. |
| 571 | Calculate the 64-bit source global address. |
| 572 | Use cache-global for 16-byte copies and cache-all for smaller supported copies. |
| 573 | Emit the direct global-to-shared async instruction with no intermediate data registers. |
| 574 | Dispatch this typed statement to the corresponding lowering path below. |
| 575 | Close the pending per-thread copy group at the IR's exact commit point. |
| 576 | Dispatch this typed statement to the corresponding lowering path below. |
| 577 | Wait for the number of outstanding groups specified by the schedule. |
| 578 | Enter the alternative path when the preceding condition does not hold. |
| 579 | Reject an unknown statement kind; no fallback renderer or compiler is invoked. |

#### compile at lines 581 to 594

```python
 581 |     def compile(self) -> str:
 582 |         self.body(self.kernel.body)
 583 |         self.emit("ret;")
 584 |         entry = symbol(self.kernel.name)
 585 |         params = ",\n".join(f"    .param .u64 param_{i}" for i in range(len(self.params)))
 586 |         header = [".version 8.0", ".target sm_89", ".address_size 64", "", *self.module_declarations,
 587 |                   f".visible .entry {entry}(\n{params}\n)", "{"]
 588 |         types = {"r": "b32", "rd": "b64", "f": "f32", "p": "pred"}
 589 |         declarations = [f"    .reg .{types[p]} %{p}<{n + 1}>;" for p, n in self.counts.items()]
 590 |         source = "\n".join(header + declarations + ["    " + d for d in self.declarations] + self.lines + ["}", ""])
 591 |         source.encode("ascii")
 592 |         return source
 593 | 
 594 | 
```

| Line | Explanation |
| --- | --- |
| 581 | Define `compile` with the parameters and result contract shown in this signature. |
| 582 | Generate the body before printing declarations so final virtual register counts are known. |
| 583 | Terminate the kernel explicitly. |
| 584 | Validate and use the original KernelOp entry name. |
| 585 | Declare one 64-bit positional pointer slot for every metadata argument. |
| 586 | Place required version, target and address-size directives first, followed by module declarations. |
| 587 | Add the visible entry signature and open the function body. |
| 588 | Map virtual register namespaces to their declaration types. |
| 589 | Declare counted register arrays, allowing the highest allocated one-based number. |
| 590 | Concatenate header, declarations, instructions, closing brace and final newline. |
| 591 | Validate the entire module as ASCII before returning it to the CLI. |
| 592 | Return complete PTX as one string so failures never leave partial output on stdout. |

#### compile_file at lines 595 to 605

```python
 595 | def compile_file(path: Path | str) -> str:
 596 |     path = Path(path)
 597 |     graph = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
 598 |     nodes = [node for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
 599 |     if len(nodes) != 1:
 600 |         raise CompileError(f"expected one KernelOp, found {len(nodes)}")
 601 |     metadata = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
 602 |     matches = [k for k in metadata["kernels"] if k["entry"] == nodes[0].op.name]
 603 |     if len(matches) != 1:
 604 |         raise CompileError("meta.json must contain exactly one matching entry")
 605 |     return Compiler(graph, nodes[0], matches[0]).compile()
```

| Line | Explanation |
| --- | --- |
| 595 | Define `compile_file` with the parameters and result contract shown in this signature. |
| 596 | Normalize string or Path input to a filesystem Path. |
| 597 | Read UTF-8 JSON and let Emmy deserialize its typed graph and statement objects. |
| 598 | Find executable KernelOp nodes without depending on a hardcoded graph node name. |
| 599 | The evaluator's per-file contract expects one kernel per input file. |
| 600 | Report zero or multiple kernels as an unsupported input layout. |
| 601 | Read the launch contract beside this specific IR file. |
| 602 | Match the metadata entry by the IR's kernel entry name, not the filename or directory. |
| 603 | Require a unique matching ABI and launch descriptor. |
| 604 | Reject absent or ambiguous metadata matches explicitly. |
| 605 | Create an isolated compiler instance and return its complete generated module. |

### tools/validate.py

Source snapshot SHA256 `69e980c854800d270aedff2ad55b689356292579cdbf628d2edff73c8ca9a4a6`.

#### Module imports and definitions at lines 1 to 27

```python
   1 | #!/usr/bin/env python3
   2 | """Assemble and optionally execute every open problem without CUDA C++ or torch.
   3 | 
   4 |     python tools/validate.py
   5 |     python tools/validate.py --assemble-only
   6 |     python tools/validate.py p00_scalar_add r01_array_sum
   7 | """
   8 | 
   9 | from __future__ import annotations
  10 | 
  11 | import argparse
  12 | import ctypes as ct
  13 | import json
  14 | import os
  15 | import subprocess
  16 | import sys
  17 | import time
  18 | from pathlib import Path
  19 | 
  20 | import numpy as np
  21 | 
  22 | ROOT = Path(__file__).resolve().parents[1]
  23 | sys.path.insert(0, str(ROOT))
  24 | 
  25 | from emmy_ptx import compile_file
  26 | 
  27 | 
```

| Line | Explanation |
| --- | --- |
| 1 | Document the adjacent implementation decision or script invocation. |
| 2 | Document the purpose and usage of this module, type or method. |
| 4 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 5 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 6 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 7 | Document the purpose and usage of this module, type or method. |
| 9 | Import the required definitions from `__future__`; no renderer is invoked here. |
| 11 | Import `argparse` for the module operations described above. |
| 12 | Import `ctypes as ct` for the module operations described above. |
| 13 | Import `json` for the module operations described above. |
| 14 | Import `os` for the module operations described above. |
| 15 | Import `subprocess` for the module operations described above. |
| 16 | Import `sys` for the module operations described above. |
| 17 | Import `time` for the module operations described above. |
| 18 | Import the required definitions from `pathlib`; no renderer is invoked here. |
| 20 | Import `numpy as np` for the module operations described above. |
| 22 | Resolve the repository root relative to this script, allowing invocation from another directory. |
| 23 | Make the root-level compiler module importable when running a tools script directly. |
| 25 | Reuse the exact same lowering implementation as compile.py. |

#### Cuda at lines 28 to 30

```python
  28 | class Cuda:
  29 |     """Small CUDA Driver API host, used only to validate the emitted cubins."""
  30 | 
```

| Line | Explanation |
| --- | --- |
| 28 | Introduce the class that groups the following state and behavior. |
| 29 | Document the purpose and usage of this module, type or method. |

#### __init__ at lines 31 to 63

```python
  31 |     def __init__(self):
  32 |         self.lib = ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
  33 |         pointer, integer, size, deviceptr = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_uint64
  34 |         signatures = {
  35 |             "cuInit": [ct.c_uint],
  36 |             "cuDeviceGet": [ct.POINTER(integer), integer],
  37 |             "cuDeviceGetName": [ct.c_char_p, integer, integer],
  38 |             "cuCtxCreate_v2": [ct.POINTER(pointer), ct.c_uint, integer],
  39 |             "cuCtxDestroy_v2": [pointer],
  40 |             "cuModuleLoadData": [ct.POINTER(pointer), pointer],
  41 |             "cuModuleUnload": [pointer],
  42 |             "cuModuleGetFunction": [ct.POINTER(pointer), pointer, ct.c_char_p],
  43 |             "cuFuncSetAttribute": [pointer, integer, integer],
  44 |             "cuMemAlloc_v2": [ct.POINTER(deviceptr), size],
  45 |             "cuMemFree_v2": [deviceptr],
  46 |             "cuMemcpyHtoD_v2": [deviceptr, pointer, size],
  47 |             "cuMemcpyDtoH_v2": [pointer, deviceptr, size],
  48 |             "cuLaunchKernel": [pointer] + [ct.c_uint] * 7 + [pointer, ct.POINTER(pointer), pointer],
  49 |             "cuCtxSynchronize": [],
  50 |             "cuGetErrorName": [integer, ct.POINTER(ct.c_char_p)],
  51 |         }
  52 |         for name, args in signatures.items():
  53 |             function = getattr(self.lib, name)
  54 |             function.argtypes, function.restype = args, integer
  55 |         self.call("cuInit", 0)
  56 |         device = integer()
  57 |         self.call("cuDeviceGet", ct.byref(device), 0)
  58 |         name = ct.create_string_buffer(256)
  59 |         self.call("cuDeviceGetName", name, len(name), device)
  60 |         self.name = name.value.decode()
  61 |         self.context = pointer()
  62 |         self.call("cuCtxCreate_v2", ct.byref(self.context), 0, device)
  63 | 
```

| Line | Explanation |
| --- | --- |
| 31 | Define `__init__` with the parameters and result contract shown in this signature. |
| 32 | Load the system NVIDIA driver library on Windows or Linux without linking a CUDA C++ program. |
| 33 | Choose fixed ctypes widths for host pointers, integers, sizes and CUDA device addresses. |
| 34 | Define Driver API signatures so Python does not truncate 64-bit pointers by default. |
| 35 | Describe the driver's initialization function and its unsigned flags argument. |
| 36 | Describe selection of a CUDA device by ordinal. |
| 37 | Describe the API that writes the selected device's display name. |
| 38 | Describe creation of an owned CUDA context using the legacy exported driver entry point. |
| 39 | Describe cleanup of that context. |
| 40 | Describe loading a cubin image from host memory into a module handle. |
| 41 | Describe releasing a previously loaded module. |
| 42 | Describe resolving a kernel entry by name within a module. |
| 43 | Describe function attribute configuration, including dynamic shared-memory opt-in. |
| 44 | Describe GPU allocation with a 64-bit device-pointer output. |
| 45 | Describe release of a device allocation. |
| 46 | Describe a host-to-device copy with a size_t byte count. |
| 47 | Describe a device-to-host copy with a size_t byte count. |
| 48 | Describe the function handle, three grid sizes, three block sizes, shared bytes, stream and parameter pointers. |
| 49 | Describe context synchronization, which takes no explicit arguments. |
| 50 | Describe conversion of a numeric driver status to a human-readable error name. |
| 51 | Complete the multiline structure or call that begins on the preceding lines. |
| 52 | Install each declared signature before making driver calls. |
| 53 | Look up the named exported entry point in the loaded library. |
| 54 | Set argument and return types to preserve the C API ABI. |
| 55 | Initialize the driver before device lookup or allocation. |
| 56 | Reserve a host integer for the chosen device handle. |
| 57 | Select device ordinal zero. |
| 58 | Allocate a writable host buffer for the GPU's name. |
| 59 | Ask the driver to fill that name buffer. |
| 60 | Decode the name for the report's hardware provenance. |
| 61 | Reserve an opaque context-handle output slot. |
| 62 | Create the context that owns all later allocations and modules. |

#### call at lines 64 to 70

```python
  64 |     def call(self, name, *args):
  65 |         status = getattr(self.lib, name)(*args)
  66 |         if status:
  67 |             message = ct.c_char_p()
  68 |             self.lib.cuGetErrorName(status, ct.byref(message))
  69 |             raise RuntimeError(f"{name}: {(message.value or str(status).encode()).decode()}")
  70 | 
```

| Line | Explanation |
| --- | --- |
| 64 | Define `call` with the parameters and result contract shown in this signature. |
| 65 | Invoke a driver call using its installed ctypes signature. |
| 66 | Treat a nonzero CUDA result as failure. |
| 67 | Reserve a C string pointer for the error name. |
| 68 | Resolve the driver error without hiding the original failing operation. |
| 69 | Raise an exception containing the operation and driver error or numeric fallback. |

#### close at lines 71 to 73

```python
  71 |     def close(self):
  72 |         self.call("cuCtxDestroy_v2", self.context)
  73 | 
```

| Line | Explanation |
| --- | --- |
| 71 | Define `close` with the parameters and result contract shown in this signature. |
| 72 | Destroy the owned context after validation finishes. |

#### run at lines 74 to 116

```python
  74 |     def run(self, cubin: Path, kernel: dict, inputs: dict) -> dict:
  75 |         module, function = ct.c_void_p(), ct.c_void_p()
  76 |         image = ct.create_string_buffer(cubin.read_bytes())
  77 |         self.call("cuModuleLoadData", ct.byref(module), image)
  78 |         allocations, arrays = [], {}
  79 |         try:
  80 |             self.call("cuModuleGetFunction", ct.byref(function), module, kernel["entry"].encode("ascii"))
  81 |             if kernel["smem_bytes"] > 48 * 1024:
  82 |                 self.call("cuFuncSetAttribute", function, 8, kernel["smem_bytes"])
  83 |             for arg in kernel["args"]:
  84 |                 name, shape = arg["name"], arg["shape"]
  85 |                 if arg.get("output") or name in kernel["zero_outputs"]:
  86 |                     array = np.zeros(shape, dtype=np.float32)
  87 |                 elif arg.get("intermediate"):
  88 |                     array = np.asarray(inputs[name], dtype=np.float32).reshape(shape)
  89 |                 elif "data" in arg:
  90 |                     array = np.asarray(arg["data"], dtype=np.float32).reshape(shape)
  91 |                 else:
  92 |                     array = np.random.default_rng(arg["seed"]).standard_normal(shape).astype(np.float32)
  93 |                 array = np.ascontiguousarray(array)
  94 |                 pointer = ct.c_uint64()
  95 |                 self.call("cuMemAlloc_v2", ct.byref(pointer), array.nbytes)
  96 |                 allocations.append(pointer)
  97 |                 self.call("cuMemcpyHtoD_v2", pointer, array.ctypes.data, array.nbytes)
  98 |                 arrays[name] = (array, pointer)
  99 |             parameters = (ct.c_void_p * len(kernel["arg_order"]))(
 100 |                 *(ct.cast(ct.byref(arrays[name][1]), ct.c_void_p) for name in kernel["arg_order"]))
 101 |             self.call("cuLaunchKernel", function, *kernel["grid"], *kernel["block"],
 102 |                       kernel["smem_bytes"], None, parameters, None)
 103 |             self.call("cuCtxSynchronize")
 104 |             outputs = {}
 105 |             for arg in kernel["args"]:
 106 |                 if arg.get("output"):
 107 |                     array, pointer = arrays[arg["name"]]
 108 |                     self.call("cuMemcpyDtoH_v2", array.ctypes.data, pointer, array.nbytes)
 109 |                     outputs[arg["name"]] = array
 110 |             return outputs
 111 |         finally:
 112 |             for pointer in allocations:
 113 |                 self.call("cuMemFree_v2", pointer)
 114 |             self.call("cuModuleUnload", module)
 115 | 
 116 | 
```

| Line | Explanation |
| --- | --- |
| 74 | Define `run` with the parameters and result contract shown in this signature. |
| 75 | Reserve separate module and kernel function handles. |
| 76 | Keep the cubin bytes alive in a C-compatible host buffer during module loading. |
| 77 | Load the assembled native binary, rather than passing CUDA source or asking NVRTC to compile. |
| 78 | Track allocations for cleanup and host-array/device-pointer pairs by argument name. |
| 79 | Guarantee module and allocation cleanup if lookup, launch or validation fails. |
| 80 | Resolve the exact metadata entry point from the cubin. |
| 81 | Detect a launch requiring more than the default dynamic shared-memory allocation. |
| 82 | Set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, whose numeric value is 8. |
| 83 | Construct every runtime argument from its metadata descriptor. |
| 84 | Retrieve the buffer name and logical shape. |
| 85 | Initialize outputs and explicitly zeroed atomic accumulators before this launch. |
| 86 | Allocate host zero data with the correct shape and float32 representation. |
| 87 | Recognize data produced by a preceding kernel in this problem. |
| 88 | Convert that intermediate output back to the next kernel's declared shape. |
| 89 | Recognize literal input data such as one-element constants. |
| 90 | Materialize the exact supplied data as a float32 array. |
| 91 | Enter the alternative path when the preceding condition does not hold. |
| 92 | Recreate the problem generator's seeded normal input, drawing in default precision before conversion. |
| 93 | Ensure the host bytes are contiguous before a direct driver copy. |
| 94 | Reserve a 64-bit device-pointer variable whose address can also become a kernel parameter. |
| 95 | Allocate exactly the host array's byte count on the GPU. |
| 96 | Record the allocation immediately so later failures do not leak it. |
| 97 | Copy the input or initialized output bytes to the GPU. |
| 98 | Keep both the host array and device-pointer variable alive under their logical name. |
| 99 | Allocate the host array of pointers to parameter storage expected by cuLaunchKernel. |
| 100 | Populate that parameter array in arg_order, independently of descriptor or graph order. |
| 101 | Launch with the prescribed grid and block dimensions. |
| 102 | Pass the prescribed dynamic shared byte count, default stream and positional parameters. |
| 103 | Wait for execution and surface asynchronous launch faults before reading outputs. |
| 104 | Collect output arrays for the reference comparison and any later kernel. |
| 105 | Visit the descriptors to identify output buffers. |
| 106 | Copy back only descriptors marked as outputs. |
| 107 | Retrieve the paired host destination and GPU allocation. |
| 108 | Read the completed device bytes into the host array. |
| 109 | Expose the resulting array under its logical graph name. |
| 110 | Return completed outputs while the finally block releases driver resources. |
| 111 | Run this cleanup on both successful completion and exceptions. |
| 112 | Visit every device allocation owned by this invocation. |
| 113 | Free that allocation through the CUDA driver. |
| 114 | Unload the module after its function and allocations are no longer needed. |

#### validate at lines 117 to 157

```python
 117 | def validate(problem: Path, output: Path, ptxas: str, arch: str, cuda: Cuda | None) -> dict:
 118 |     metadata = json.loads((problem / "meta.json").read_text())
 119 |     produced, records = {}, []
 120 |     for kernel in metadata["kernels"]:
 121 |         ir = problem / (kernel["entry"] + ".ir.json")
 122 |         start = time.perf_counter()
 123 |         source = compile_file(ir)
 124 |         lower_ms = (time.perf_counter() - start) * 1000
 125 |         ptx = output / (problem.name + "_" + kernel["entry"] + ".ptx")
 126 |         ptx.write_text(source, encoding="ascii")
 127 |         cubin = ptx.with_suffix(".cubin")
 128 |         start = time.perf_counter()
 129 |         result = subprocess.run([ptxas, f"-arch={arch}", "-v", str(ptx), "-o", str(cubin)],
 130 |                                 capture_output=True, text=True, timeout=30)
 131 |         assembly_ms = (time.perf_counter() - start) * 1000
 132 |         if result.returncode:
 133 |             raise RuntimeError(result.stderr or result.stdout or f"ptxas exited with {result.returncode}")
 134 |         record = {"entry": kernel["entry"], "lower_ms": lower_ms, "assembly_ms": assembly_ms,
 135 |                   "ptx_bytes": len(source), "ptxas": (result.stdout + result.stderr).strip(), "outputs": {}}
 136 |         if cuda is not None:
 137 |             produced.update(cuda.run(cubin, kernel, produced))
 138 |             with np.load(problem / "expected.npz") as expected:
 139 |                 for name in expected.files:
 140 |                     if name not in produced:
 141 |                         continue
 142 |                     reference = expected[name]
 143 |                     actual = produced[name].reshape(reference.shape)
 144 |                     error = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
 145 |                     passed = actual.shape == reference.shape and np.allclose(
 146 |                         actual, reference, rtol=metadata["rtol"], atol=metadata["atol"])
 147 |                     record["outputs"][name] = {"pass": bool(passed), "max_abs_error": error}
 148 |                     if not passed:
 149 |                         raise AssertionError(f"{problem.name}/{name}: max absolute error {error}")
 150 |         records.append(record)
 151 |     if cuda is not None:
 152 |         missing = set(metadata["graph_outputs"]) - produced.keys()
 153 |         if missing:
 154 |             raise AssertionError(f"missing graph outputs: {sorted(missing)}")
 155 |     return {"problem": problem.name, "status": "PASS" if cuda else "ASSEMBLED", "kernels": records}
 156 | 
 157 | 
```

| Line | Explanation |
| --- | --- |
| 117 | Define `validate` with the parameters and result contract shown in this signature. |
| 118 | Read this problem's launch and numeric comparison contract. |
| 119 | Track chained outputs and per-kernel evidence separately. |
| 120 | Respect the metadata's kernel order for possible intermediate dependencies. |
| 121 | Select the IR file named by the kernel entry. |
| 122 | Start an in-process lowering wall-clock measurement. |
| 123 | Generate PTX through the production compilation path. |
| 124 | Record lowering time, including IR deserialization, in milliseconds. |
| 125 | Use both problem and entry name to avoid output artifact collisions. |
| 126 | Write ASCII PTX for ptxas and later human inspection. |
| 127 | Choose the matching native-binary artifact path. |
| 128 | Start timing the separate assembly subprocess. |
| 129 | Invoke only ptxas, with architecture, verbose output and explicit cubin path. |
| 130 | Capture both output streams and enforce a 30-second assembler limit. |
| 131 | Record assembly wall time independently of lowering. |
| 132 | Check the assembler's process status rather than assuming a file was produced correctly. |
| 133 | Include stdout as a diagnostic fallback because Windows ptxas can report errors there. |
| 134 | Record the kernel entry and both compilation-stage measurements. |
| 135 | Record source size, assembler output and the output-comparison container. |
| 136 | Skip all device execution in assembly-only mode. |
| 137 | Execute the cubin and merge outputs for multi-kernel problems. |
| 138 | Open the reference archive with automatic cleanup. |
| 139 | Visit every expected named output. |
| 140 | Do not compare an output before its producing kernel has executed. |
| 141 | Continue until that output becomes available. |
| 142 | Read the reference array with its original scalar or tensor shape. |
| 143 | Match the official harness by reshaping the returned buffer to the reference shape. |
| 144 | Calculate a readable maximum absolute error, including the empty-output case. |
| 145 | Check shape agreement and start the numeric tolerance comparison. |
| 146 | Use the problem's own rtol and atol rather than one global tolerance. |
| 147 | Convert NumPy's predicate to a JSON-compatible bool and store the error. |
| 148 | Detect a failed output while still identifying the specific problem and buffer. |
| 149 | Raise a correctness failure containing its maximum error. |
| 150 | Keep the completed per-kernel evidence for this problem. |
| 151 | Check output completeness only when kernels were actually executed. |
| 152 | Find graph outputs absent from the produced-buffer map. |
| 153 | Detect missing required outputs even if no comparison ran for them. |
| 154 | Fail rather than treating an incomplete execution as a pass. |
| 155 | Distinguish an execution PASS from successful assembly without a GPU run. |

#### main at lines 158 to 187

```python
 158 | def main() -> int:
 159 |     parser = argparse.ArgumentParser(description=__doc__)
 160 |     parser.add_argument("problems", nargs="*")
 161 |     parser.add_argument("--assemble-only", action="store_true")
 162 |     parser.add_argument("--ptxas", default="ptxas")
 163 |     parser.add_argument("--arch", default="sm_89")
 164 |     parser.add_argument("--output", type=Path, default=ROOT / "build" / "validation")
 165 |     args = parser.parse_args()
 166 |     problems = [ROOT / "problems" / name for name in args.problems] if args.problems else sorted(
 167 |         path.parent for path in (ROOT / "problems").glob("*/meta.json"))
 168 |     args.output.mkdir(parents=True, exist_ok=True)
 169 |     cuda = None if args.assemble_only else Cuda()
 170 |     report = {"gpu": cuda.name if cuda else None, "arch": args.arch, "results": []}
 171 |     try:
 172 |         for problem in problems:
 173 |             try:
 174 |                 result = validate(problem, args.output, args.ptxas, args.arch, cuda)
 175 |             except Exception as exc:
 176 |                 result = {"problem": problem.name, "status": "FAIL", "error": str(exc)}
 177 |             report["results"].append(result)
 178 |             print(f"{problem.name}: {result['status']}" + (f" {result['error']}" if "error" in result else ""), flush=True)
 179 |     finally:
 180 |         if cuda is not None:
 181 |             cuda.close()
 182 |     (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
 183 |     return int(any(r["status"] == "FAIL" for r in report["results"]))
 184 | 
 185 | 
 186 | if __name__ == "__main__":
 187 |     sys.exit(main())
```

| Line | Explanation |
| --- | --- |
| 158 | Define `main` with the parameters and result contract shown in this signature. |
| 159 | Expose local validation modes through a small CLI. |
| 160 | Accept an optional subset of problem directory names. |
| 161 | Allow assembly checks on a machine without a GPU. |
| 162 | Allow an explicit ptxas executable path when it is not on PATH. |
| 163 | Default local cubin generation to the scheduled architecture. |
| 164 | Keep generated binaries and evidence in the ignored build tree by default. |
| 165 | Parse the selected mode, problems and output destination. |
| 166 | Use explicitly named problems when supplied, otherwise discover public packages. |
| 167 | Identify package directories by meta.json rather than by naming conventions. |
| 168 | Create the artifact destination, including its parent build directory. |
| 169 | Create a CUDA context only for execution mode. |
| 170 | Record hardware and architecture alongside per-problem results. |
| 171 | Enter an operation whose expected errors or resource cleanup are handled below. |
| 172 | Validate the chosen problems in order. |
| 173 | Enter an operation whose expected errors or resource cleanup are handled below. |
| 174 | Run the entire lowering, assembly and optional execution path for one problem. |
| 175 | Convert a problem-specific exception into a reportable failure. |
| 176 | Preserve the problem name and exception text in the result record. |
| 177 | Append either success or failure evidence to the report. |
| 178 | Print a compact progress/result line immediately, including diagnostics on failure. |
| 179 | Run this cleanup on both successful completion and exceptions. |
| 180 | Only destroy a context if execution mode created one. |
| 181 | Release the driver's context even when iteration raises an exception. |
| 182 | Persist structured evidence with readable indentation and a final newline. |
| 183 | Return failure if any problem failed while still reporting all attempted problems. |
| 186 | Run the validation CLI only for direct script execution. |
| 187 | Expose main's aggregate result as the process status. |

### tests/test_compiler.py

Source snapshot SHA256 `e13db40887680fa8103b0d45a70e797e979d60004fcc84767b0f086e0e3a9e8c`.

#### Module imports and definitions at lines 1 to 31

```python
   1 | """Semantic regressions on fresh typed IRs, with optional real GPU execution.
   2 | 
   3 | Run CPU checks with python -m unittest discover -s tests -v.
   4 | Set EMMY_PTX_GPU_TESTS=1 to assemble and execute the synthetic cases too.
   5 | """
   6 | 
   7 | import json
   8 | import os
   9 | import subprocess
  10 | import sys
  11 | import tempfile
  12 | import unittest
  13 | from pathlib import Path
  14 | 
  15 | import numpy as np
  16 | from emmy.compiler.dim import Dim
  17 | from emmy.compiler.dtype import F32
  18 | from emmy.compiler.graph import Graph, Node
  19 | from emmy.compiler.ir.axis import Axis
  20 | from emmy.compiler.ir.base import InputOp
  21 | from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
  22 | from emmy.compiler.ir.kernel.ir import KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle
  23 | from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write
  24 | from emmy.compiler.tensor import Tensor
  25 | 
  26 | from emmy_ptx import CompileError, compile_file, constant
  27 | from tools.validate import Cuda, validate
  28 | 
  29 | ROOT = Path(__file__).resolve().parents[1]
  30 | 
  31 | 
```

| Line | Explanation |
| --- | --- |
| 1 | Document the purpose and usage of this module, type or method. |
| 3 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 4 | Continue the expression or declaration begun on the preceding line; its result is used by that statement. |
| 5 | Document the purpose and usage of this module, type or method. |
| 7 | Import `json` for the module operations described above. |
| 8 | Import `os` for the module operations described above. |
| 9 | Import `subprocess` for the module operations described above. |
| 10 | Import `sys` for the module operations described above. |
| 11 | Import `tempfile` for the module operations described above. |
| 12 | Import `unittest` for the module operations described above. |
| 13 | Import the required definitions from `pathlib`; no renderer is invoked here. |
| 15 | Import `numpy as np` for the module operations described above. |
| 16 | Import the required definitions from `emmy.compiler.dim`; no renderer is invoked here. |
| 17 | Import the required definitions from `emmy.compiler.dtype`; no renderer is invoked here. |
| 18 | Import the required definitions from `emmy.compiler.graph`; no renderer is invoked here. |
| 19 | Import the required definitions from `emmy.compiler.ir.axis`; no renderer is invoked here. |
| 20 | Import the required definitions from `emmy.compiler.ir.base`; no renderer is invoked here. |
| 21 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 22 | Import the required definitions from `emmy.compiler.ir.kernel.ir`; no renderer is invoked here. |
| 23 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 24 | Import the required definitions from `emmy.compiler.tensor`; no renderer is invoked here. |
| 26 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 27 | Import the required definitions from `tools.validate`; no renderer is invoked here. |
| 29 | Resolve repository paths independently of the temporary fixture location. |

#### axis at lines 32 to 35

```python
  32 | def axis(name, extent):
  33 |     return Axis(name, Dim(extent))
  34 | 
  35 | 
```

| Line | Explanation |
| --- | --- |
| 32 | Define `axis` with the parameters and result contract shown in this signature. |
| 33 | Construct a typed static axis without tracing any torch expression. |

#### binary at lines 36 to 39

```python
  36 | def binary(op, left, right):
  37 |     return BinaryExpr(op, left, Literal(right, "int"))
  38 | 
  39 | 
```

| Line | Explanation |
| --- | --- |
| 36 | Define `binary` with the parameters and result contract shown in this signature. |
| 37 | Construct a typed binary expression with an integer literal right operand. |

#### CompilerTests at lines 40 to 40

```python
  40 | class CompilerTests(unittest.TestCase):
```

| Line | Explanation |
| --- | --- |
| 40 | Introduce the class that groups the following state and behavior. |

#### setUp at lines 41 to 44

```python
  41 |     def setUp(self):
  42 |         self.temporary = tempfile.TemporaryDirectory()
  43 |         self.folder = Path(self.temporary.name)
  44 | 
```

| Line | Explanation |
| --- | --- |
| 41 | Define `setUp` with the parameters and result contract shown in this signature. |
| 42 | Give every test its own isolated fixture directory. |
| 43 | Represent the temporary fixture root as a Path. |

#### tearDown at lines 45 to 47

```python
  45 |     def tearDown(self):
  46 |         self.temporary.cleanup()
  47 | 
```

| Line | Explanation |
| --- | --- |
| 45 | Define `tearDown` with the parameters and result contract shown in this signature. |
| 46 | Remove only the test-owned temporary directory after the test. |

#### fixture at lines 48 to 70

```python
  48 |     def fixture(self, inputs, output_shape, body, expected, block, grid=1):
  49 |         graph = Graph()
  50 |         graph.inputs, graph.outputs = list(inputs), ["answer"]
  51 |         descriptors = []
  52 |         for name, data in inputs.items():
  53 |             data = np.asarray(data, dtype=np.float32)
  54 |             tensor = Tensor(name, tuple(Dim(d) for d in data.shape), F32)
  55 |             graph.nodes[name] = Node(name, InputOp(), [], (tensor,))
  56 |             descriptors.append({"name": name, "shape": list(data.shape), "dtype": "f32", "data": data.ravel().tolist()})
  57 |         output = Tensor("answer", tuple(Dim(d) for d in output_shape), F32)
  58 |         graph.nodes["answer"] = Node("answer", KernelOp(body=(body,), name="fresh_kernel"), list(inputs), (output,))
  59 |         descriptors.append({"name": "answer", "shape": list(output_shape), "dtype": "f32", "output": True})
  60 |         # Outputs first and reversed inputs exercise the positional ABI independently of graph order.
  61 |         descriptors.reverse()
  62 |         kernel = {"entry": "fresh_kernel", "arg_order": [a["name"] for a in descriptors], "args": descriptors,
  63 |                   "block": [block, 1, 1], "grid": [grid, 1, 1], "smem_bytes": 0, "zero_outputs": []}
  64 |         metadata = {"kernels": [kernel], "graph_outputs": ["answer"], "rtol": 1e-5, "atol": 1e-5}
  65 |         path = self.folder / "fresh_kernel.ir.json"
  66 |         path.write_text(json.dumps(graph.to_dict(), default=str), encoding="utf-8")
  67 |         (self.folder / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
  68 |         np.savez(self.folder / "expected.npz", answer=np.asarray(expected, dtype=np.float32))
  69 |         return path
  70 | 
```

| Line | Explanation |
| --- | --- |
| 48 | Define `fixture` with the parameters and result contract shown in this signature. |
| 49 | Start a fresh graph rather than copying a public answer kernel. |
| 50 | Declare graph inputs and the single expected output. |
| 51 | Collect the argument descriptors that will form metadata's ABI. |
| 52 | Build typed input tensors from the supplied arrays. |
| 53 | Normalize synthetic input values to float32. |
| 54 | Convert concrete array dimensions to Emmy Dim objects. |
| 55 | Add an InputOp node producing the typed tensor. |
| 56 | Store exact flattened input data so both compiler and reference are independent of seeds. |
| 57 | Create the output tensor's static shape and dtype. |
| 58 | Wrap the newly constructed statement tree in a newly named KernelOp. |
| 59 | Mark the output argument as runtime-allocated storage. |
| 60 | Explain why the descriptor order deliberately differs from graph order. |
| 61 | Put the output first and reverse input order to expose incorrect positional ABI assumptions. |
| 62 | Use this reversed order as the explicit arg_order contract. |
| 63 | Define a launch with no dynamic shared storage and no atomic zeroing requirements. |
| 64 | Set expected graph outputs and numeric tolerances for the synthetic case. |
| 65 | Choose a fresh entry filename rather than one of the open fixtures. |
| 66 | Serialize through Emmy's graph schema, using representations for typed body objects. |
| 67 | Write the matching metadata next to the IR, as required by compile_file. |
| 68 | Write an independent NumPy expected-output archive for the local validator. |
| 69 | Return the fixture path to the selected test. |

#### check_fixture at lines 71 to 82

```python
  71 |     def check_fixture(self, path):
  72 |         source = compile_file(path)
  73 |         self.assertTrue(source.isascii())
  74 |         self.assertTrue(source.startswith(".version 8.0\n"))
  75 |         if os.environ.get("EMMY_PTX_GPU_TESTS") == "1":
  76 |             cuda = Cuda()
  77 |             try:
  78 |                 result = validate(self.folder, self.folder, "ptxas", "sm_89", cuda)
  79 |                 self.assertEqual(result["status"], "PASS")
  80 |             finally:
  81 |                 cuda.close()
  82 | 
```

| Line | Explanation |
| --- | --- |
| 71 | Define `check_fixture` with the parameters and result contract shown in this signature. |
| 72 | Lower the newly generated fixture using the production API. |
| 73 | Assert that every output character is permitted in PTX. |
| 74 | Assert the module starts with the configured PTX version directive. |
| 75 | Keep device execution explicitly opt-in for environments without a GPU. |
| 76 | Create a real CUDA Driver API context for semantic validation. |
| 77 | Enter an operation whose expected errors or resource cleanup are handled below. |
| 78 | Assemble and run the synthetic program against its independent expected result. |
| 79 | Require an execution PASS rather than merely successful assembly. |
| 80 | Run this cleanup on both successful completion and exceptions. |
| 81 | Destroy the test's context even if assembly or a comparison fails. |

#### test_vector_loads_nondivisible_tile_and_reordered_arguments at lines 83 to 94

```python
  83 |     def test_vector_loads_nondivisible_tile_and_reordered_arguments(self):
  84 |         left = np.arange(140, dtype=np.float32).reshape(7, 20)
  85 |         right = np.full((7, 20), 17, dtype=np.float32)
  86 |         index = (Var("row"), binary("*", Var("col"), 4))
  87 |         body = [Load(input="left", index=index, names=tuple(f"l{i}" for i in range(4)), dtype=F32),
  88 |                 Load(input="right", index=index, names=tuple(f"r{i}" for i in range(4)), dtype=F32)]
  89 |         body += [Assign(f"v{i}", "subtract", (f"l{i}", f"r{i}"), F32) for i in range(4)]
  90 |         body += [Write("answer", index, values=tuple(f"v{i}" for i in range(4)), value_dtype=F32)]
  91 |         tile = Tile((axis("row", 7), axis("col", 5)), tuple(body))
  92 |         path = self.fixture({"left": left, "right": right}, (7, 20), tile, left-right, block=32, grid=2)
  93 |         self.check_fixture(path)
  94 | 
```

| Line | Explanation |
| --- | --- |
| 83 | Define `test_vector_loads_nondivisible_tile_and_reordered_arguments` with the parameters and result contract shown in this signature. |
| 84 | Choose a rectangular tensor whose vectorized cell count does not fill the launch. |
| 85 | Use a distinct constant right operand so subtraction also detects argument-order errors. |
| 86 | Address four contiguous columns per logical cell. |
| 87 | Create the first genuine vector load with four destination names. |
| 88 | Create a second vector load from the independently passed right input. |
| 89 | Compute one subtraction per loaded component with operation names absent from this public pointwise fixture. |
| 90 | Write the four results as one grouped vector store. |
| 91 | Use a two-axis tile whose total of 35 cells is below the 64 launched threads. |
| 92 | Supply an independently calculated NumPy subtraction and reversed ABI metadata. |
| 93 | Check text generation and, with the flag enabled, real GPU correctness. |

#### test_maximum_reduction_with_empty_thread_partials at lines 95 to 109

```python
  95 |     def test_maximum_reduction_with_empty_thread_partials(self):
  96 |         data = -np.arange(1, 186, dtype=np.float32).reshape(5, 37)
  97 |         combine = (Assign("acc", "maximum", ("acc", "other"), F32),)
  98 |         loop = StridedLoop(axis("j", 37), Var("co"), Literal(64, "int"), (
  99 |             Load("x", "input", (Var("row"), Var("j")), dtype=F32),
 100 |             Accum("acc", "x", "maximum", F32, ("j",)),))
 101 |         tile = Tile((axis("row", 5), axis("co", 64)), (
 102 |             loop, WarpShuffle(("acc",), ("other",), combine, 32), Smem("partials", (2,)),
 103 |             Cond(binary("==", Var("lane"), 0), (Write("partials", (Var("warp"),), "acc", value_dtype=F32),)),
 104 |             Sync(), TreeHalve(("partials",), ("acc",), ("other",), combine, 2, "warp"),
 105 |             Cond(binary("==", Var("co"), 0), (Write("answer", (Var("row"),), "acc", value_dtype=F32),)),
 106 |         ), block_threads=64)
 107 |         path = self.fixture({"input": data}, (5,), tile, data.max(axis=1), block=64, grid=5)
 108 |         self.check_fixture(path)
 109 | 
```

| Line | Explanation |
| --- | --- |
| 95 | Define `test_maximum_reduction_with_empty_thread_partials` with the parameters and result contract shown in this signature. |
| 96 | Use all-negative values so an incorrect zero identity for maximum is detectable. |
| 97 | Define the actual maximum combine program rather than assuming addition. |
| 98 | Give 64 cooperative threads only 37 elements per row, creating empty partials. |
| 99 | Load the current row and strided element using typed expressions. |
| 100 | Reduce values with maximum, whose identity must come from operation metadata. |
| 101 | Create five independent rows with one cooperative block per row. |
| 102 | Combine within each warp and declare one shared slot per warp. |
| 103 | Store only each warp's lane-zero partial. |
| 104 | Synchronize and combine the two shared warp partials. |
| 105 | Let only one cooperative coordinate store the row's completed result. |
| 106 | Preserve the explicit 64-thread cooperative compute band. |
| 107 | Compare the GPU's maximum with NumPy's independent row maximum. |
| 108 | Run the common compile and optional GPU checks. |

#### test_seed_false_keeps_accumulator_across_nested_loops at lines 110 to 118

```python
 110 |     def test_seed_false_keeps_accumulator_across_nested_loops(self):
 111 |         data = np.arange(11, dtype=np.float32)
 112 |         inner = Loop(axis("inner", 3), (Accum("acc", "x", "add", F32),), seed=False)
 113 |         outer = Loop(axis("outer", 2), (inner,))
 114 |         tile = Tile((axis("i", 11),), (Load("x", "input", (Var("i"),), dtype=F32),
 115 |                     Init("acc", 0, dtype=F32), outer, Write("answer", (Var("i"),), "acc", value_dtype=F32)))
 116 |         path = self.fixture({"input": data}, (11,), tile, data*6, block=32)
 117 |         self.check_fixture(path)
 118 | 
```

| Line | Explanation |
| --- | --- |
| 110 | Define `test_seed_false_keeps_accumulator_across_nested_loops` with the parameters and result contract shown in this signature. |
| 111 | Use an 11-cell tile to exercise an additional partial launch. |
| 112 | Prevent the inner loop from resetting the accumulator on every outer iteration. |
| 113 | Repeat the inner three additions twice through a separate enclosing loop. |
| 114 | Load one value per useful thread cell. |
| 115 | Initialize the accumulator once before both loops and write its final value afterward. |
| 116 | Require six copies of the input, exposing a mistaken inner reseed that would yield three. |
| 117 | Check generation and optional native execution of the nested-loop fixture. |

#### test_cli_is_ptx_only_and_reports_errors_on_stderr at lines 119 to 133

```python
 119 |     def test_cli_is_ptx_only_and_reports_errors_on_stderr(self):
 120 |         data = np.arange(3, dtype=np.float32)
 121 |         tile = Tile((axis("i", 3),), (Load("x", "input", (Var("i"),), dtype=F32),
 122 |                     Write("answer", (Var("i"),), "x", value_dtype=F32)))
 123 |         path = self.fixture({"input": data}, (3,), tile, data, block=32)
 124 |         result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)], capture_output=True, text=True)
 125 |         self.assertEqual(result.returncode, 0, result.stderr)
 126 |         self.assertEqual(result.stderr, "")
 127 |         self.assertEqual(result.stdout, compile_file(path))
 128 |         (self.folder / "meta.json").unlink()
 129 |         result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)], capture_output=True, text=True)
 130 |         self.assertNotEqual(result.returncode, 0)
 131 |         self.assertEqual(result.stdout, "")
 132 |         self.assertIn("meta.json", result.stderr)
 133 | 
```

| Line | Explanation |
| --- | --- |
| 119 | Define `test_cli_is_ptx_only_and_reports_errors_on_stderr` with the parameters and result contract shown in this signature. |
| 120 | Use a tiny identity-copy input to isolate command-line behavior from arithmetic. |
| 121 | Construct a typed load for each useful cell. |
| 122 | Write that loaded value to the synthetic output. |
| 123 | Create valid adjacent IR and metadata before invoking the command. |
| 124 | Invoke the exact submission entry point in a separate Python process. |
| 125 | Require a successful exit and include stderr in any assertion diagnostic. |
| 126 | Require no diagnostic output on a valid invocation. |
| 127 | Require stdout to equal the backend's complete PTX string exactly. |
| 128 | Remove metadata to create a clear, deterministic input-contract failure. |
| 129 | Repeat the same command against the now-incomplete package. |
| 130 | Require failure rather than output from a guessed ABI. |
| 131 | Ensure that failed compilation did not print a partial PTX module. |
| 132 | Require a useful diagnostic naming the missing metadata file. |

#### test_rejects_unsupported_math_without_partial_ptx at lines 134 to 141

```python
 134 |     def test_rejects_unsupported_math_without_partial_ptx(self):
 135 |         data = np.ones(1, dtype=np.float32)
 136 |         tile = Tile((axis("i", 1),), (Load("x", "input", (Var("i"),), dtype=F32),
 137 |                     Assign("y", "sin", ("x",), F32), Write("answer", (Var("i"),), "y", value_dtype=F32)))
 138 |         path = self.fixture({"input": data}, (1,), tile, np.sin(data), block=32)
 139 |         with self.assertRaisesRegex(CompileError, "unsupported elementwise"):
 140 |             compile_file(path)
 141 | 
```

| Line | Explanation |
| --- | --- |
| 134 | Define `test_rejects_unsupported_math_without_partial_ptx` with the parameters and result contract shown in this signature. |
| 135 | Choose simple data so the failure is caused by unsupported lowering, not input shape. |
| 136 | Create the valid load preceding the unsupported operation. |
| 137 | Insert a registered Emmy operation intentionally absent from this backend's supported math. |
| 138 | Serialize the unsupported program using the normal fixture path. |
| 139 | Require the backend's explicit unsupported-operation exception. |
| 140 | Trigger compilation inside the exception assertion. |

#### test_nested_loop_local_does_not_overwrite_outer_value at lines 142 to 149

Reproduce the old scope bug by shadowing a float local in a nested loop and requiring the outer value afterward.

```python
 142 |     def test_nested_loop_local_does_not_overwrite_outer_value(self):
 143 |         tile = Tile((axis("i", 3),), (
 144 |             Let("saved", Literal(2.0), F32),
 145 |             Loop(axis("j", 2), (Let("saved", Literal(9.0), F32),)),
 146 |             Write("answer", (Var("i"),), "saved", value_dtype=F32)))
 147 |         path = self.fixture({}, (3,), tile, np.full(3, 2.0), block=32)
 148 |         self.check_fixture(path)
 149 | 
```

| Line | Explanation |
| --- | --- |
| 142 | Reproduce the old scope bug by shadowing a float local in a nested loop and requiring the outer value afterward. |
| 143 | Compute and bind `tile` for the surrounding operation described in this function. |
| 144 | Compute and bind `tile` for the surrounding operation described in this function. |
| 145 | Compute and bind `tile` for the surrounding operation described in this function. |
| 146 | Compute and bind `tile` for the surrounding operation described in this function. |
| 147 | Compute and bind `path` for the surrounding operation described in this function. |
| 148 | Generate the fixture PTX and, when GPU tests are enabled, assemble and execute it against the independent reference. |

#### test_signed_index_division_and_modulo at lines 150 to 161

Exercise negative intermediate integer coordinates for all three floor-division/modulo spellings on a real GPU.

```python
 150 |     def test_signed_index_division_and_modulo(self):
 151 |         data = np.arange(12, dtype=np.float32)
 152 |         negative = binary("-", Var("i"), 5)
 153 |         for divisor in (3, -3):
 154 |             for op in ("/", "//", "%"):
 155 |                 index = binary("+", binary(op, negative, divisor), 3)
 156 |                 expected_index = ((np.arange(8)-5) % divisor if op == "%" else (np.arange(8)-5)//divisor)+3
 157 |                 tile = Tile((axis("i", 8),), (Load("x", "input", (index,), dtype=F32),
 158 |                             Write("answer", (Var("i"),), "x", value_dtype=F32)))
 159 |                 path = self.fixture({"input": data}, (8,), tile, data[expected_index], block=32)
 160 |                 self.check_fixture(path)
 161 | 
```

| Line | Explanation |
| --- | --- |
| 150 | Exercise negative intermediate integer coordinates for all three floor-division/modulo spellings on a real GPU. |
| 151 | Compute and bind `data` for the surrounding operation described in this function. |
| 152 | Compute and bind `negative` for the surrounding operation described in this function. |
| 153 | Visit the specified elements in order, retaining the surrounding scope and error handling. |
| 154 | Visit the specified elements in order, retaining the surrounding scope and error handling. |
| 155 | Compute and bind `index` for the surrounding operation described in this function. |
| 156 | Compute and bind `expected_index` for the surrounding operation described in this function. |
| 157 | Compute and bind `tile` for the surrounding operation described in this function. |
| 158 | Compute and bind `tile` for the surrounding operation described in this function. |
| 159 | Compute and bind `path` for the surrounding operation described in this function. |
| 160 | Generate the fixture PTX and, when GPU tests are enabled, assemble and execute it against the independent reference. |

#### test_predicate_comparison_and_predicate_ternary at lines 162 to 172

Exercise nested predicate equality and predicate selection, which previously failed ptxas assembly.

```python
 162 |     def test_predicate_comparison_and_predicate_ternary(self):
 163 |         less = binary("<", Var("i"), 2)
 164 |         greater = binary(">", Var("i"), 0)
 165 |         equal = BinaryExpr("==", less, greater)
 166 |         choice = TernaryExpr(less, equal, greater)
 167 |         value = TernaryExpr(choice, Literal(1.0), Literal(-1.0))
 168 |         tile = Tile((axis("i", 4),), (Let("x", value, F32),
 169 |                     Write("answer", (Var("i"),), "x", value_dtype=F32)))
 170 |         path = self.fixture({}, (4,), tile, [-1, 1, 1, 1], block=32)
 171 |         self.check_fixture(path)
 172 | 
```

| Line | Explanation |
| --- | --- |
| 162 | Exercise nested predicate equality and predicate selection, which previously failed ptxas assembly. |
| 163 | Compute and bind `less` for the surrounding operation described in this function. |
| 164 | Compute and bind `greater` for the surrounding operation described in this function. |
| 165 | Compute and bind `equal` for the surrounding operation described in this function. |
| 166 | Compute and bind `choice` for the surrounding operation described in this function. |
| 167 | Compute and bind `value` for the surrounding operation described in this function. |
| 168 | Compute and bind `tile` for the surrounding operation described in this function. |
| 169 | Compute and bind `tile` for the surrounding operation described in this function. |
| 170 | Compute and bind `path` for the surrounding operation described in this function. |
| 171 | Generate the fixture PTX and, when GPU tests are enabled, assemble and execute it against the independent reference. |

#### test_let_honors_declared_f32_dtype_for_integer_literal at lines 173 to 178

Require a typed integer literal to have numeric float32 value after a declared conversion.

```python
 173 |     def test_let_honors_declared_f32_dtype_for_integer_literal(self):
 174 |         tile = Tile((axis("i", 3),), (Let("x", Literal(2, "int"), F32),
 175 |                     Write("answer", (Var("i"),), "x", value_dtype=F32)))
 176 |         path = self.fixture({}, (3,), tile, np.full(3, 2.0), block=32)
 177 |         self.check_fixture(path)
 178 | 
```

| Line | Explanation |
| --- | --- |
| 173 | Require a typed integer literal to have numeric float32 value after a declared conversion. |
| 174 | Compute and bind `tile` for the surrounding operation described in this function. |
| 175 | Compute and bind `tile` for the surrounding operation described in this function. |
| 176 | Compute and bind `path` for the surrounding operation described in this function. |
| 177 | Generate the fixture PTX and, when GPU tests are enabled, assemble and execute it against the independent reference. |

#### test_rejects_nonpositive_static_stride at lines 179 to 185

Reject a statically zero stride without launching an infinite loop.

```python
 179 |     def test_rejects_nonpositive_static_stride(self):
 180 |         tile = Tile((axis("i", 1),), (StridedLoop(axis("j", 2), Literal(0, "int"), Literal(0, "int"), ()),
 181 |                     Let("x", Literal(1.0), F32), Write("answer", (Var("i"),), "x", value_dtype=F32)))
 182 |         path = self.fixture({}, (1,), tile, [1], block=32)
 183 |         with self.assertRaisesRegex(CompileError, "positive"):
 184 |             compile_file(path)
 185 | 
```

| Line | Explanation |
| --- | --- |
| 179 | Reject a statically zero stride without launching an infinite loop. |
| 180 | Compute and bind `tile` for the surrounding operation described in this function. |
| 181 | Compute and bind `tile` for the surrounding operation described in this function. |
| 182 | Compute and bind `path` for the surrounding operation described in this function. |
| 183 | Require an explicit compiler error rather than silently generating incorrect PTX. |
| 184 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |

#### test_rejects_mismatched_reduction_components at lines 186 to 192

Reject malformed state lengths instead of silently dropping components through zip.

```python
 186 |     def test_rejects_mismatched_reduction_components(self):
 187 |         tile = Tile((axis("i", 32),), (Init("acc", 0, dtype=F32),
 188 |                     WarpShuffle(("acc",), (), (), 32),))
 189 |         path = self.fixture({}, (1,), tile, [0], block=32)
 190 |         with self.assertRaisesRegex(CompileError, "component"):
 191 |             compile_file(path)
 192 | 
```

| Line | Explanation |
| --- | --- |
| 186 | Reject malformed state lengths instead of silently dropping components through zip. |
| 187 | Compute and bind `tile` for the surrounding operation described in this function. |
| 188 | Compute and bind `tile` for the surrounding operation described in this function. |
| 189 | Compute and bind `path` for the surrounding operation described in this function. |
| 190 | Require an explicit compiler error rather than silently generating incorrect PTX. |
| 191 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |

#### test_worker_recovers_after_an_error_and_matches_single_file_compilation at lines 193 to 208

Send an invalid request followed by two valid requests and compare their complete output to one-shot compilation.

```python
 193 |     def test_worker_recovers_after_an_error_and_matches_single_file_compilation(self):
 194 |         tile = Tile((axis("i", 2),), (Let("x", Literal(7.0), F32),
 195 |                     Write("answer", (Var("i"),), "x", value_dtype=F32)))
 196 |         path = self.fixture({}, (2,), tile, [7, 7], block=32)
 197 |         requests = ["not json", json.dumps({"ir": str(path)}), json.dumps({"ir": str(path)})]
 198 |         result = subprocess.run([sys.executable, str(ROOT / "compile.py"), "--worker"],
 199 |                                 input="\n".join(requests)+"\n", capture_output=True, text=True, timeout=10)
 200 |         self.assertEqual(result.returncode, 0, result.stderr)
 201 |         self.assertEqual(result.stderr, "")
 202 |         responses = [json.loads(line) for line in result.stdout.splitlines()]
 203 |         self.assertEqual(len(responses), 3)
 204 |         self.assertFalse(responses[0]["ok"])
 205 |         for response in responses[1:]:
 206 |             self.assertTrue(response["ok"])
 207 |             self.assertEqual(response["ptx"], compile_file(path))
 208 | 
```

| Line | Explanation |
| --- | --- |
| 193 | Send an invalid request followed by two valid requests and compare their complete output to one-shot compilation. |
| 194 | Compute and bind `tile` for the surrounding operation described in this function. |
| 195 | Compute and bind `tile` for the surrounding operation described in this function. |
| 196 | Compute and bind `path` for the surrounding operation described in this function. |
| 197 | Compute and bind `requests` for the surrounding operation described in this function. |
| 198 | Compute and bind `result` for the surrounding operation described in this function. |
| 199 | Compute and bind `result` for the surrounding operation described in this function. |
| 200 | Require the stated behavior as an executable regression assertion. |
| 201 | Require the stated behavior as an executable regression assertion. |
| 202 | Compute and bind `responses` for the surrounding operation described in this function. |
| 203 | Require the stated behavior as an executable regression assertion. |
| 204 | Require the stated behavior as an executable regression assertion. |
| 205 | Visit the specified elements in order, retaining the surrounding scope and error handling. |
| 206 | Require the stated behavior as an executable regression assertion. |
| 207 | Require the stated behavior as an executable regression assertion. |

#### test_literal_cache_preserves_signed_zero at lines 209 to 215

Decode emitted float bit patterns independently and check that negative zero retains its sign bit.

```python
 209 |     def test_literal_cache_preserves_signed_zero(self):
 210 |         import struct
 211 |         positive = constant(0.0).text[2:]
 212 |         negative = constant(-0.0).text[2:]
 213 |         self.assertEqual(struct.unpack("!f", bytes.fromhex(positive))[0], 0.0)
 214 |         self.assertTrue(np.signbit(struct.unpack("!f", bytes.fromhex(negative))[0]))
 215 | 
```

| Line | Explanation |
| --- | --- |
| 209 | Decode emitted float bit patterns independently and check that negative zero retains its sign bit. |
| 210 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 211 | Compute and bind `positive` for the surrounding operation described in this function. |
| 212 | Compute and bind `negative` for the surrounding operation described in this function. |
| 213 | Require the stated behavior as an executable regression assertion. |
| 214 | Require the stated behavior as an executable regression assertion. |

#### test_rejects_integer_register_in_float_store at lines 216 to 226

Require an explicit type error instead of a bitwise integer-to-float misinterpretation.

```python
 216 |     def test_rejects_integer_register_in_float_store(self):
 217 |         from emmy.compiler.dtype import I32
 218 |         tile = Tile((axis("i", 1),), (Let("x", Literal(7, "int"), I32),
 219 |                     Write("answer", (Var("i"),), "x", value_dtype=F32)))
 220 |         path = self.fixture({}, (1,), tile, [7], block=32)
 221 |         with self.assertRaisesRegex(CompileError, "float32 stores"):
 222 |             compile_file(path)
 223 | 
 224 | 
 225 | if __name__ == "__main__":
 226 |     unittest.main()
```

| Line | Explanation |
| --- | --- |
| 216 | Require an explicit type error instead of a bitwise integer-to-float misinterpretation. |
| 217 | Import the module or typed definitions used by this file; this line does not perform lowering. |
| 218 | Compute and bind `tile` for the surrounding operation described in this function. |
| 219 | Compute and bind `tile` for the surrounding operation described in this function. |
| 220 | Compute and bind `path` for the surrounding operation described in this function. |
| 221 | Require an explicit compiler error rather than silently generating incorrect PTX. |
| 222 | Continue the surrounding statement or literal; its role is explained by this function and the preceding operation. |
| 225 | Allow direct unittest execution in addition to discovery mode. |
| 226 | Invoke unittest to run this module when it is executed directly. |
