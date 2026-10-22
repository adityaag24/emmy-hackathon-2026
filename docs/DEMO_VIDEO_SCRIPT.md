# Emmy direct PTX compiler demo script

This script is designed for a 4 minute 30 second video, within the README's
3 to 5 minute requirement. Explain the code in your own voice and adjust the
wording to match your understanding. The recorded results below were verified
on October 22, 2026; rerun the commands before recording.

## Before recording

Use Python 3.12 or newer with `python -m pip install -r requirements.txt`. Activate
your environment so `python` resolves to it. Open the README, `compile.py`,
`emmy_ptx.py`, the reduction IR text and the generated PTX in your editor.

Generate the example with:

```bash
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
ptxas -arch=sm_89 out.ptx -o out.cubin
python submit.py r01_array_sum --compiler compile.py
```

The hosted command needs network access. Keep a recent output visible in case
the service is busy, and clearly label a prerecorded result. With CUDA tools and
an NVIDIA GPU, `python tools/validate.py` runs all 15 problems locally and writes
`build/validation/report.json`. The video does not require a local GPU.

## 0 minutes to 35 seconds The problem

**Show:** the README and this two-path diagram, either on screen or on a slide.

```text
Existing: Emmy Kernel IR -> CUDA C++ -> nvcc -> native GPU binary
Ours:     Emmy Kernel IR -> our PTX compiler -> ptxas -> native GPU binary
```

**Say:**

“This project is a direct PTX backend for Emmy. Emmy compiles PyTorch expressions
into scheduled Kernel IR. Its usual path generates CUDA C++ and then invokes
nvcc. During autotuning, it can compile thousands of kernel variants, so that
compilation step matters.

Our compiler takes the existing Kernel IR and emits PTX directly. NVIDIA's
ptxas assembler then produces a native GPU binary. The hackathon README reports
about a nineteen-times assembly advantage for that shorter path. Our own
benchmarks separate fresh-process invocation from persistent compilation; the
measured results are in the repository's benchmark report.”

## 35 seconds to 75 seconds The input and interface

**Show:** the reduction's `.ir.txt`, then its `meta.json`, then `compile.py`.

**Say:**

“The input is a typed program, not a problem name. This reduction has a Tile,
a strided loop, an accumulator, a warp shuffle, shared-memory scratch, a barrier,
and a final write.

We load it with Emmy's Graph.from_dict. All translation after loading is our own.
We do not call Emmy's CUDA renderer or nvcc.

The kernel entry name comes from the IR. Parameter order comes from meta.json,
because the evaluator passes arguments positionally. Constants are also buffer
arguments. The command-line wrapper writes only PTX to standard output and sends
errors to standard error.”

## 75 seconds to 135 seconds How the compiler works

**Show:** `Compiler.tile`, `address`, `body`, and `compile` in `emmy_ptx.py`.

**Say:**

“The backend walks the statement tree and maps IR values to typed virtual
registers. It declares the register counts after generating the body; ptxas does
the physical register allocation.

For a Tile, we derive a global thread index, reject indices outside the tile,
and decode the axis coordinates. That guard is essential when a launch has more
threads than useful cells.

For memory, multidimensional coordinates become a row-major offset. We multiply
by four for float32, then add a 64-bit global pointer or a shared-memory base.
Grouped values become vector loads or stores.

Arithmetic and comparisons become PTX instructions. Loops become labels and
branches. The compiler follows the IR structure, so a new shape or kernel name
does not need a new template.”

## 135 seconds to 185 seconds The difficult cases

**Show:** `loop`, `shuffle`, `tree`, and the async-copy cases in `body`.

**Say:**

“Accumulator placement is the key to nested loops. A seeded loop initializes
its own reduction state before entering. A loop marked seed false preserves
state initialized outside it.

Warp reductions use XOR shuffles. Shared-memory reductions combine the warp
partials with a barrier after every tree level, then broadcast the final value.
Softmax has coupled state: both a maximum and a rescaled sum. We lower the IR's
actual combine program rather than assuming every reduction is addition.

The matmul path also emits asynchronous global-to-shared copies and their
commit and wait instructions. Large shared allocations use the dynamic buffer
specified by the launch. Atomic output updates remain atomic adds.”

## 185 seconds to 235 seconds Run the demo

**Show:** run the three preflight commands. Open `out.ptx` near the shuffle and
`bar.sync` instructions. Show the `PASS` from the runner. Then show the all-problem
local validation output or clearly labeled captured official results.

**Say:**

“Here the required command generates PTX from the reduction IR. The output starts
with the PTX version, architecture and address-size directives, followed by the
correct entry point. These shuffle and barrier instructions came directly from
the reduction statements.

ptxas accepts the file and produces a cubin. The official submit script compiles
and runs our compiler's output, then compares it with the expected NumPy result
under the prescribed launch and tolerance.

All fifteen open problems passed on both our local Ada GPU and the official
RTX 5090 runner. That includes pointwise operations, reductions, both matmuls,
fused normalization, row softmax and both attention problems.”

## 235 seconds to 270 seconds Evidence and scope

**Show:** `tests/test_compiler.py`, then the validation and limitations sections
in `docs/PROJECT_WALKTHROUGH.md`.

**Say:**

“We also constructed new IR tests for reordered arguments, vector loads, a tile
that does not fill the launch, maximum reduction with empty thread partials,
and nested loops that preserve an accumulator. Those GPU tests pass too.

The current scope is static float32 kernels. It does not implement tensor cores,
TMA or software shared-memory swizzles. We have verified the open set, not the
hidden evaluation set. The output has the same native binary format as the
usual toolchain; it is not claimed to be byte-identical.

The result is a working compiler that removes CUDA C++ generation from this
path and leaves native code generation to ptxas.”

## Recording cues

Pause briefly after each command so the output is readable. Keep the terminal
font large and navigate directly to named functions. Aim for about 125 to 140
spoken words per minute; shorten the code-navigation pauses if the recording
approaches five minutes. Do not describe runner latency as a kernel benchmark,
or describe the README's nineteen-times figure as a measurement from this code.
The measured comparison is documented in `docs/REVIEW_AND_BENCHMARK.md`, which
separates fresh-process invocation from persistent compilation.

