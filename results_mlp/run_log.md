# Run log — real model layer (T5)

Interpreter for every command:
`/hpc2ssd/softwares/anaconda3/envs/pytorch_gpu_2.0.1/bin/python`
(python 3.11.4, torch 2.0.1+cu118, torchvision 0.15.2+cu118, numpy 1.24.1;
CPU only, `torch.set_num_threads(1)` inside the program).
Working directory: `recovery-baseline/`. No timestamps are recorded by
design; full raw captures of every step are kept in `results_mlp/logs/`.

## 1. Environment probe + shuffle-RNG source inspection

```
$ python -c "import torch, torchvision, numpy, sys; print(torch.__version__,
  torchvision.__version__, numpy.__version__, sys.version); print(torch.get_num_threads())"
2.0.1+cu118
0.15.2+cu118
1.24.1
3.11.4 (main, Jul  5 2023, 13:45:01) [GCC 11.2.0]
64            # default thread count -> forced to 1 by the program

$ python -c "import inspect; from torch.utils.data import sampler;
  print(inspect.getsource(sampler.RandomSampler.__iter__))"
# -> with an explicit generator and replacement=False, __iter__ consumes the
#    passed generator once per epoch via torch.randperm(n, generator=gen).
#    This is why the shuffle RNG is checkpointable and why a mid-epoch
#    checkpoint needs the epoch-start rewind (see equivalence.md).
```

## 2. MNIST download into ./data_mlp

```
$ python - <<'EOF'
import torchvision
ds = torchvision.datasets.MNIST(root='./data_mlp', train=True, download=True)
ts = torchvision.datasets.MNIST(root='./data_mlp', train=False, download=True)
print('train', len(ds), 'test', len(ts))
EOF
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next): HTTP Error 404: Not Found
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz ... (OK)
... (same pattern for the other 3 files; all 4 fetched from the
     ossci-datasets S3 fallback, extracted successfully)
train 60000 test 10000
```

MNIST downloaded cleanly. The deprecated yann.lecun.com mirror 404s and
torchvision falls back to the ossci-datasets S3 mirror, exactly as the
environment recon predicted; no synthetic-data fallback was needed or used.

## 3. Smoke test of the `run` CLI

```
$ python train_mlp.py run --epochs 1
{"epochs": [{"epoch": 1, "train_loss": 1.428710401058197, "lr": 0.05}], "step": 32, "test_accuracy": 0.82}
exit=0
```

32 batches/epoch as designed (2000/64, drop_last=False); 3 epochs = 96 steps.

## 4. Main experiment: compare

First attempt FAILED on the mid-epoch split (this is a recorded failure
case, not a silent fix):

```
$ python train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47
ERROR: checkpoint results_mlp/checkpoints/ckpt_batch_47.pt has inconsistent
epoch/step (epoch=2, step=47, batches_per_epoch=32); refusing to continue
exit=2
```

Two real bugs found and fixed before the successful run:

1. **Epoch-field semantics at a mid-epoch stop.** A partial epoch advanced
   the checkpoint's `epoch` field and called `scheduler.step()`, so a
   mid-epoch checkpoint claimed 2 epochs done at step 47 and carried a
   scheduler that had stepped twice — while the uninterrupted worldline at
   step 47 has scheduler stepped once. Fixed: `epoch` = fully completed
   epochs (`step // batches_per_epoch`); scheduler steps only when an epoch
   completes fully.
2. **RandomSampler re-draw (the interesting one).** The naive mid-epoch
   checkpoint captured the shuffle generator AFTER the current epoch's
   randperm had been consumed. On resume, DataLoader iteration draws a
   FRESH randperm (torch 2.0.1 `RandomSampler.__iter__`), so the resumed
   run skipped batches of the NEXT epoch's permutation — the batch stream
   diverged from the reference immediately (observed max per-step loss diff
   0.44 right after the split; tier (b) not bit-exact). Fixed: snapshot the
   shuffle generator at each epoch start (a pure read, no RNG consumed) and
   rewind it if the epoch stops partway, so the checkpointed generator
   always sits at an epoch boundary of the draw sequence.

Successful run (full captured output in `logs/compare.stdout`, report in
`results_mlp/equivalence.md`):

```
$ python train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47
exit=0
# -> (b) full-state resume BIT-EXACT (True) at BOTH the epoch boundary and
#       the mid-epoch split: max param diff 0.0, allclose True, per-step
#       loss diff 0, final test acc 0.882 identical to reference.
# -> (c) fresh optimizer/scheduler and (d) fresh all-RNG diverge far above
#       tolerance (param diffs 0.026-0.036; acc deltas -0.020..+0.014).
```

## 5. Cross-process CLI check (run -> checkpoint -> resume in new processes)

```
$ python train_mlp.py run --epochs 1 --out results_mlp/checkpoints/cli_epoch1.pt
{"epochs": [{"epoch": 1, "train_loss": 1.428710401058197, "lr": 0.05}], "step": 32, "test_accuracy": 0.82, "checkpoint_saved": "results_mlp/checkpoints/cli_epoch1.pt"}

$ python train_mlp.py run --epochs 3 --out results_mlp/checkpoints/cli_full.pt
{"epochs": [{"epoch": 1, "train_loss": 1.428710401058197, "lr": 0.05}, {"epoch": 2, "train_loss": 0.4963734280318022, "lr": 0.04000000000000001}, {"epoch": 3, "train_loss": 0.36271160608157516, "lr": 0.03200000000000001}], "step": 96, "test_accuracy": 0.882, "checkpoint_saved": "results_mlp/checkpoints/cli_full.pt"}

$ python train_mlp.py run --epochs 3 --resume results_mlp/checkpoints/cli_epoch1.pt
{"epochs": [{"epoch": 2, "train_loss": 0.4963734280318022, "lr": 0.04000000000000001}, {"epoch": 3, "train_loss": 0.36271160608157516, "lr": 0.03200000000000001}], "step": 96, "test_accuracy": 0.882}
```

Cross-process full-state resume reproduces the uninterrupted run's epochs
2-3 loss values bit-for-bit (identical printed floats) and the same final
test accuracy 0.882.

## 6. Guard behaviors (refusing-to semantics, matching the toy layer)

```
$ python train_mlp.py run --epochs 1 --resume results_mlp/checkpoints/cli_full.pt
ERROR: requested epochs 1 is behind checkpoint step 96 (refusing to go backwards)
exit=2

$ python train_mlp.py run --epochs 1 --resume results_mlp/checkpoints/cli_epoch1.pt
{"epochs": [], "step": 32, "test_accuracy": 0.82}     # idempotent no-op, allowed
exit=0
```

## 7. Negative tests (selftest)

```
$ python train_mlp.py selftest
PASS: missing file (exit=2, stderr=nonempty, traceback=no)
PASS: truncated/corrupt checkpoint (exit=2, stderr=nonempty, traceback=no)
PASS: wrong format_version (exit=2, stderr=nonempty, traceback=no)
PASS: missing required key (optimizer) (exit=2, stderr=nonempty, traceback=no)
PASS: field 'epoch' has wrong type (str) (exit=2, stderr=nonempty, traceback=no)
exit=0
```

Full per-case stderr capture in `results_mlp/negative_tests.txt`.

## Known limitations

- Single fixed software stack (python 3.11.4 / torch 2.0.1+cu118 /
  torchvision 0.15.2 / numpy 1.24.1, 64-thread box forced to 1 thread).
  Cross-version `rng_state` compatibility untested by design.
- All `compare` evidence is produced by in-process runs (identical fixed
  seeds); the cross-process check (section 5) covers resume semantics via
  the CLI but compares printed metrics, not per-step trajectories.
- Toy-sized task: 96 steps, 2000 train / 500 test samples; accuracy deltas
  are quantized at 1 sample = 0.002. Divergence magnitudes of partial
  recovery tiers say nothing about real workloads.
- No GPU, no multithread, no DataLoader workers, no augmentation/dropout —
  the four RNG sources saved are all the RNG consumers this model has.
- Mid-epoch resume relies on the epoch-start rewind of the shuffle
  generator; a checkpoint written by a DIFFERENT framework version or a
  loader with `replacement=True`/multi-worker semantics would need
  re-validation.
- The bit-exact result for tier (b) is an observation under the switches
  above, recorded per REPORT.md's pre-written boundary; it is not a
  guarantee and is not claimed as one.
