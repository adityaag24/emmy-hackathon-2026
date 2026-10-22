# Emmy Hackathon — write a PTX compiler for Kernel IR

## Implemented compiler

This repository now includes a direct Kernel IR to PTX compiler. It uses Emmy only
to deserialize typed IR, and implements the lowering in [emmy_ptx.py](emmy_ptx.py).
The submission entry point is [compile.py](compile.py).

```bash
python -m pip install -r requirements.txt
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
ptxas -arch=sm_89 out.ptx -o out.cubin
python submit.py r01_array_sum --compiler compile.py
```

Keep `meta.json` beside the input IR: it supplies the positional parameter order
and launch descriptor. The compiler writes ASCII PTX 8.0 targeting `sm_89` and
requires no GPU or CUDA installation to generate that text. `ptxas` is required
for assembly; a GPU is required for execution.

**Verified on October 22, 2026:** all 15 open problems pass both on a local NVIDIA
RTX 3500 Ada Generation Laptop GPU and through the official RTX 5090 runner using
`submit.py --compiler compile.py`. This covers all five categories. Hidden-set
correctness has not been tested. See the [review and benchmark report](docs/REVIEW_AND_BENCHMARK.md)
for repeated measurements against the public CUDA Toolkit 13.4 `nvcc`.

```bash
python -m unittest discover -s tests -v
python tools/validate.py --assemble-only  # needs ptxas, no GPU
python tools/validate.py                  # needs ptxas and an NVIDIA GPU
```

The local validator uses the CUDA Driver API directly, with no CUDA C++ or torch.
Set `EMMY_PTX_GPU_TESTS=1` when running the unit tests to execute the additional
synthetic IR cases on a GPU. Generated PTX, cubins and test reports go into ignored
`build/` directories.

Supported data is static-shape `f32` with one-dimensional launches. The backend
handles scalar and vector memory operations, arithmetic, loops, conditionals,
reductions, shared memory, asynchronous copies, atomics and online softmax state.
Raster hints use an equivalent row-major CTA order; loop unroll hints retain
runtime loops. Tensor cores, TMA, software shared-memory swizzles, dynamic shapes,
by-value scalars and serial runtime axes are outside this implementation's scope.

- [Demo video script](docs/DEMO_VIDEO_SCRIPT.md): a four-and-a-half-minute recording plan.
- [Project and line-by-line code walkthrough](docs/PROJECT_WALKTHROUGH.md): architecture,
  lowering details, source annotations, validation evidence and limitations.
- [Code review and compilation benchmark](docs/REVIEW_AND_BENCHMARK.md): reproduced
  defects, fixes, timing methodology, measured results and reproduction commands.

For autotuning, `python compile.py --worker` accepts JSON-line requests such as
`{"ir":"path/to/kernel.ir.json"}` and returns `{"ok":true,"ptx":"..."}` per
request. Keeping that process alive amortizes Python and Emmy imports. Python
callers can instead import `compile_file` from `emmy_ptx` once. The required
single-file command continues to emit plain PTX and includes fresh-process
startup costs. The existing `submit.py` uses that single-file path.

```bash
python tools/benchmark.py --repetitions 5 --warmups 1 --verify
```

## Original hackathon brief

Emmy (github.com/cloudrift-ai/emmy) compiles PyTorch to CUDA C++ and builds it with `nvcc`.
**Your task: write a compiler that takes Emmy's Kernel IR and emits PTX directly**, skipping the
CUDA C++ step. `ptxas` turns PTX into the same GPU binary ~19× faster than the full `nvcc`
pipeline — Emmy's autotuner compiles thousands of kernel variants, so that middle step matters.

You do **not** need a GPU. A hosted runner at `http://runner.riftstack.ai:48701` (the default in `submit.py`) compiles
your PTX with `ptxas`, executes it on real hardware, and tells you whether the output is correct. See [docs/API.md](docs/API.md).

## Requirements

- **Python 3.12 or newer.** The Kernel IR only deserializes through Emmy's own classes:

  ```bash
  pip install emmy-ml==0.3.11       # no torch, no CUDA, no GPU needed
  ```

  This alone reads every problem IR. Torch is **not** needed — it is only required to
  *generate* new IRs from PyTorch expressions, which is optional.

- Your submission is a **program**, not PTX files. It must run as:

  ```bash
  python compile.py path/to/kernel.ir.json > out.ptx
  ```

  This contract matters: submissions are evaluated against a **hidden set** of Kernel IRs in the
  same categories as the open problems. Hardcoding the open set scores nothing.

## Five-minute start

```bash
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 && cd emmy-hackathon-2026
pip install emmy-ml==0.3.11 numpy requests   # no torch needed
python submit.py r01_array_sum examples/k_sum_1_reduce.ptx    # see a pass
python submit.py p00_scalar_add examples/k_sum_1_reduce.ptx   # wrong entry: a clean failure
python submit.py p00_scalar_add --compiler compile.py         # your compiler, once you have one
```

Then open `problems/p00_scalar_add/` and look at three files:

| file | what it is |
|---|---|
| `*.ir.json` | the Kernel IR — **the input to your compiler** |
| `*.ir.txt` | the same IR, pretty-printed for reading |
| `*.ref.cu` | Emmy's generated CUDA C++ — the reference answer, in a different target language |

Load an IR in Python:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/r01_array_sum/k_sum_1_reduce.ir.json")))
body = g.nodes["sum_1"].op.body     # a tree of typed statements: Tile, StridedLoop, Load, ...
```

Your compiler walks that tree and prints PTX. Start with the common node kinds in the short table —
see [TASK.md](TASK.md) for the categories and the node-by-node translation table.

## The problems

15 open problems in 5 categories, easy → hard: **pointwise → reductions → matmul → fused →
attention-like**. Each is a folder under `problems/` with the IR, the reference, a launch
descriptor (`meta.json` — you never guess grid/block), and expected outputs (`expected.npz`).

**MVP: your compiler passes categories 1 and 2.** See [MVP.md](MVP.md).

Problems are scheduled for `sm_89`, and the runner is an **RTX 5090 (`sm_120`)**, where PTX written for `sm_89`
runs unchanged; [problems/PROVENANCE.md](problems/PROVENANCE.md)
says how they were generated and checked. The hidden evaluation set contains more problems of the *same kinds* —
different shapes and ops, no surprise concepts.

## Submitting

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit) before
**23:59 AoE on Tuesday, October 27**. A submission is:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works — an entry where the understanding seems to
have been handed to an AI is not eligible for a prize.

## How compilers are evaluated

We run your `compile.py` on every kernel of the **hidden set** — the same five categories as the
open problems, with different shapes and ops — then assemble the PTX with `ptxas`, run it on the
runner's GPU under each problem's `meta.json` launch, and compare the outputs with the expected
ones within the problem's tolerance. This is the pipeline `submit.py --compiler` runs, so passing
the open set that way is the best preparation. Keep compiling to seconds per kernel: a time
limit may apply.

1. **Correctness on the hidden set, by category.** Passing categories 1 and 2 is the bar
   ([MVP.md](MVP.md)); each further category is what sets submissions apart.
2. **Code review** of your compiler is the first tiebreak. The IR → PTX lowering has to be your
   own — see the rule in [TASK.md](TASK.md).
3. **Speed** of the generated kernels is the last tiebreak.

The video decides prize eligibility, as described above.

## Repo map

```
problems/     the open set (15 problems)
examples/     hand-written PTX with comments mapping each block back to the IR
submit.py     one-command test: compile + run + check against expected outputs
docs/API.md   the runner API, if you want to call it directly
tools/        how the problems were generated (emmy compile -c ... --ir cuda)
TASK.md       the task, the categories, the CLI contract
MVP.md        the minimum bar
```
