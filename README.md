# DASE7506 MP1 — Small Language Model Challenge

This repository contains the code, experimental record, and report for my
DASE7506 Mini-Project 1 submission. It builds on the course-provided starter
package. The original course instructions are preserved in
[`GUIDE.md`](GUIDE.md), and the original code documentation is preserved in
[`code/README.md`](code/README.md).

## Final result

The frozen predictor was selected using the validation split and evaluated on
the full test split only after all architecture, checkpoint, ensemble-weight,
and temperature choices had been fixed.

| Split | BPB | Token perplexity |
|---|---:|---:|
| Validation | 1.5558912737 | 26.7767 |
| Full test | **1.5774969803** | 27.0494 |

The submitted full-test value is **1.577497 BPB**. Lower is better.

The report is available at
[`report/7506_mp1_report.pdf`](report/7506_mp1_report.pdf).

## Frozen predictor

The final predictor contains two independently trained SwiGLU Transformers:

- D6: width 192, depth 6, 6 attention heads, residual dropout 0.1;
- D8: width 192, depth 8, 6 attention heads, residual dropout 0.1.

Both components were trained for 7,200 updates with seed 17 on the supplied
training split. The D8 component is an equal parameter average of checkpoints
from steps 6,000, 6,600, and 7,200 of one continuous training run. At inference
time, the predictor averages probabilities with weights 0.40 for D6 and 0.60
for D8, then applies temperature scaling with `T = 1.05`.

The final checkpoint is self-contained: it stores both component models and
does not require the parent checkpoints or retraining for evaluation.

## Environment and installation

The submitted experiments used Python 3.12.14 and the following pinned
dependencies:

- PyTorch 2.7.1
- NumPy 2.5.3
- tokenizers 0.21.4

Create an environment from the repository root:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The reported runs used CPU FP32 with four PyTorch threads on an Apple M5
MacBook Air running macOS 27.0. No GPU, MPS, external dataset, pretrained
weights, API key, or evaluation-time network access is required.

## Direct evaluation without retraining

Download the separately supplied checkpoint bundle and run the following from
`code/`:

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --device cpu \
  --precision fp32 \
  --threads 4 \
  --split test \
  --output reproduced_test_cpu_fp32.json
```

The expected result is approximately:

```text
bpb = 1.5774969802531538
targets = 428405
utf8_bytes = 1292013
protocol = 7506-mp1-wt2-v2
```

Small runtime differences across machines are expected. The supplied frozen
checkpoint, rather than retraining, is the authoritative reproduction path.

## Checkpoint ancestry

The final checkpoint has SHA-256:

```text
9dc1dd4f3981407ffaaaed620e2424de7f8cf70df0c30aa225fa6885f21d9c47
```

Its two direct components are the D6 best checkpoint from step 7,200 and the
equal average of the D8 checkpoints from steps 6,000, 6,600, and 7,200.

The D8 average was constructed in two equivalent stages:

1. average steps 6,000 and 6,600 with weights 0.5/0.5;
2. combine that result with step 7,200 using weights 2/3 and 1/3.

This gives an equal one-third contribution from each of the three checkpoints.
The final ensemble stores the D6 and D8 weights directly and records the parent
hashes for provenance only.

## Optional training reproduction

Retraining is not required to reproduce the submitted score. The following
commands document the training and assembly procedure. Output directories must
not already exist.

Train D6:

```bash
python train.py \
  --implementation student \
  --config configs/swiglu-medium-dropout010.json \
  --device cpu \
  --precision fp32 \
  --threads 4 \
  --steps 7200 \
  --seed 17 \
  --eval-every 600 \
  --run-dir reproduction_runs/d6
```

Train D8 and save periodic checkpoints:

```bash
python train.py \
  --implementation student \
  --config configs/swiglu-deep-dropout010.json \
  --device cpu \
  --precision fp32 \
  --threads 4 \
  --steps 7200 \
  --seed 17 \
  --eval-every 600 \
  --save-every 600 \
  --run-dir reproduction_runs/d8
```

Construct the equal D8 late-checkpoint average:

```bash
python average_checkpoints.py \
  --checkpoint-a reproduction_runs/d8/checkpoint_step_6000.pt \
  --checkpoint-b reproduction_runs/d8/checkpoint_step_6600.pt \
  --weight-b 0.5 \
  --shared-ancestry \
  --output reproduction_runs/d8-avg-6000-6600/checkpoint.pt

python average_checkpoints.py \
  --checkpoint-a reproduction_runs/d8-avg-6000-6600/checkpoint.pt \
  --checkpoint-b reproduction_runs/d8/checkpoint_step_7200.pt \
  --weight-b 0.3333333333333333 \
  --shared-ancestry \
  --output reproduction_runs/d8-avg-6000-6600-7200/checkpoint.pt
```

Construct the probability ensemble and apply temperature scaling:

```bash
python build_ensemble_checkpoint.py \
  --checkpoint-a reproduction_runs/d6/best_checkpoint.pt \
  --checkpoint-b reproduction_runs/d8-avg-6000-6600-7200/checkpoint.pt \
  --weight-b 0.60 \
  --temperature 1.05 \
  --output reproduction_runs/final-ensemble/checkpoint.pt
```

Floating-point and system-level differences may prevent independently trained
models from being bitwise identical across platforms. Use the frozen checkpoint
for exact score reproduction.

## Resource compliance

All measurements below used the same machine, CPU FP32, four threads, and the
validation split for the timing comparison.

| Measurement | Result | Limit |
|---|---:|---:|
| Baseline median scoring time, 3 runs | 3.608 s | — |
| Final median scoring time, 3 runs | 13.306 s | — |
| Relative CPU scoring time | 3.69x | 5x |
| Final predictor peak RSS | 1.71 GiB | 4 GiB |
| Inference assets | 27.23 MiB | 64 MiB |

The full-test scoring pass took 14.854 seconds on the same machine.

The supplied data, tokenizer, `common.py`, and `evaluate.py` were not modified.

## Repository contents

- `code/model.py`: course baseline GPT;
- `code/student.py`: SwiGLU model with configurable residual dropout;
- `code/ensemble.py`: probability ensemble and temperature scaling;
- `code/train.py`: training, validation, best-checkpoint, and periodic-checkpoint logic;
- `code/average_checkpoints.py`: parameter averaging with shared-ancestry accounting;
- `code/build_ensemble_checkpoint.py`: self-contained ensemble checkpoint builder;
- `code/configs/`: baseline and final component configurations;
- `code/tests/`: supplied correctness tests;
- `code/data/`: supplied data and tokenizer;
- `report/`: final report;
- `results/`: selected evaluation and resource evidence.

Large development runs, intermediate checkpoints, virtual environments, and
cache files are intentionally excluded from version control.

## AI Assistance Disclosure

This project builds on the course-provided MP1 starter code. AI tools were used
selectively to support discussions of experimental options, review of small
code changes, troubleshooting, verification of recorded results, and
improvement of the report and repository documentation. The author determined
the experimental direction, implemented and executed the experiments,
evaluated the models, interpreted the results, and made all final technical
and presentation decisions.

All experiments were executed locally by the author. Suggested code and
commands were manually reviewed and applied by the author, and the author
verified the resulting checkpoints, evaluation outputs, resource measurements,
and reported values. The author remains responsible for understanding the
implementation and for the accuracy of the submitted code, analysis, report,
and results.
