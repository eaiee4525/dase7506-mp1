# Resource measurements

All timing measurements used the validation split, CPU FP32, four PyTorch
threads, and the same Apple M5 MacBook Air.

## CPU scoring time

Baseline measurements:

```text
3.6075422499561682 s
3.5675975829944946 s
3.7204630839987660 s
median = 3.6075422499561682 s
```

Final predictor measurements:

```text
13.306147374969441 s
13.043304124963470 s
13.393612166983075 s
median = 13.306147374969441 s
```

The median time ratio is `13.306147374969441 / 3.6075422499561682 = 3.69x`,
below the `5x` limit. The individual evaluator outputs are stored in
`results/timing/`.

## Peak memory

The final predictor was measured with macOS `/usr/bin/time -l`:

```text
maximum resident set size = 1,837,432,832 bytes = 1.71 GiB
```

This is below the 4 GiB limit. The complete command output is stored in
`results/resource/final_validation_memory_rss.txt`.

## Inference assets

```text
checkpoint.pt = 28,542,174 bytes
checkpoint.pt + ensemble.py + student.py + model.py = 28,549,522 bytes
approximately 27.23 MiB
```

This is below the 64 MiB limit.

## Full-test evaluation

The frozen predictor achieved `1.5774969802531538` BPB. The full-test scoring
pass took `14.853570624953136` seconds. The evaluator output is stored in
`results/final_test_cpu_fp32.json`.
