# Equivalence (real model layer, T5): interrupted/resumed vs uninterrupted

Model: MLP 784->128 (ReLU)->10, CrossEntropy; MNIST subsets train=2000 test=500; batch=64; SGD(lr=0.05, momentum=0.9) + StepLR(step_size=1, gamma=0.8); epochs=3 (32 batches/epoch, 96 steps total).

Determinism switches ALL ON: torch.use_deterministic_algorithms(True), torch.manual_seed, numpy.random.seed, random.seed, torch.set_num_threads(1). No op errored under deterministic mode for this model (MLP + cross_entropy on CPU).

Checkpoint = full state: model/optimizer/scheduler state_dict, epoch (completed), step (batches consumed), and all four RNG sources (torch global, numpy, python, DataLoader shuffle generator). Subset-selection RNG is separate and runs identically in every process (see train_mlp.py docstring).

Reference: final test accuracy = 0.882, per-epoch train loss = {1: '1.42871', 2: '0.496373', 3: '0.362712'}

## Split at epoch boundary after epoch 1 (global step 32)

pre-split segment (steps 1..32) bit-identical to reference: True

| tier | max abs param diff | allclose (default tol) | bit-exact | final test acc | Δacc vs ref | max per-step loss |diff| |
|---|---|---|---|---|---|---|
| b: full-state resume | 0.0 | True | True | 0.882 | +0 | 0 |
| c: fresh optimizer/scheduler | 0.03574979677796364 | False | False | 0.87 | -0.012 | 0.150365 |
| d: fresh all RNG | 0.036041975021362305 | False | False | 0.89 | +0.008 | 0.431856 |

per-epoch train-loss |diff| vs reference (means over the merged trajectory):

| tier | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| b: full-state resume | 0 | 0 | 0 |
| c: fresh optimizer/scheduler | 0 | 0.0192418 | 0.00176193 |
| d: fresh all RNG | 0 | 0.0182494 | 0.00929491 |

**OBSERVED bit-exactness of full-state resume (b) at this split: True**

## Split at mid-epoch at global batch 47 (epoch 2, batch 15 of that epoch)

pre-split segment (steps 1..47) bit-identical to reference: True

| tier | max abs param diff | allclose (default tol) | bit-exact | final test acc | Δacc vs ref | max per-step loss |diff| |
|---|---|---|---|---|---|---|
| b: full-state resume | 0.0 | True | True | 0.882 | +0 | 0 |
| c: fresh optimizer/scheduler | 0.02648003399372101 | False | False | 0.862 | -0.02 | 0.100318 |
| d: fresh all RNG | 0.029409974813461304 | False | False | 0.896 | +0.014 | 0.357167 |

per-epoch train-loss |diff| vs reference (means over the merged trajectory):

| tier | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| b: full-state resume | 0 | 0 | 0 |
| c: fresh optimizer/scheduler | 0 | 0.00565939 | 0.0161504 |
| d: fresh all RNG | 0 | 0.0154461 | 0.00228479 |

**OBSERVED bit-exactness of full-state resume (b) at this split: True**

## Bit-exactness observation for (b) and its explanation

Observed: epoch boundary after epoch 1 (global step 32): True; mid-epoch at global batch 47 (epoch 2, batch 15 of that epoch): True. Conclusions are framed at tolerance/statistical level per the pre-written boundary in REPORT.md; the bit-exact result is recorded as an observation under one fixed software stack, not as a guarantee.

Why it holds here: (i) single-thread CPU (thread scheduling cannot change float reduction order); (ii) torch.use_deterministic_algorithms(True) — no op in this model (linear, relu, cross_entropy) errored or has a nondeterministic CPU implementation; (iii) the checkpoint carries every piece of state that influences the future trajectory — weights, SGD momentum, scheduler epoch counter, and all four RNG streams, including the DataLoader shuffle generator; (iv) before each resume every RNG source was deliberately clobbered (seed 999), so the match comes from the saved state, not from lucky seed reuse.

One non-obvious requirement found during this work: torch 2.0.1's RandomSampler draws a FRESH randperm from the generator every time iteration over the DataLoader starts (RandomSampler.__iter__). A naive mid-epoch checkpoint — generator state captured after the current epoch's permutation was already drawn — therefore resumes with the NEXT epoch's permutation and is NOT bit-exact (observed during development: max per-step loss diff 0.44 immediately after a mid-epoch resume, and epoch/step inconsistency at the epoch field). The implementation snapshots the shuffle generator at each epoch start and rewinds it when a run stops mid-epoch, so the checkpointed generator always sits at an epoch boundary of the draw sequence and the current epoch's batch order is reproduced exactly. This failure mode is exactly the class of bug the toy layer's 'no RNG draws inside save/load' rule guards against, one level down in the framework.

## Honest boundary for the real model layer (T5)

Established here, with run evidence: on CPU, single-threaded, all determinism switches on, within ONE fixed software stack (python 3.11.4, torch 2.0.1+cu118, torchvision 0.15.2, numpy 1.24.1), a full-state checkpoint resume reproduces the uninterrupted run at least at tolerance level (torch.allclose default) — measured here as bit-exact at both an epoch boundary and a mid-epoch split, in-process and cross-process. Also measured: the cost of each partial-recovery tier. Fresh optimizer/scheduler, or fresh RNG with optimizer intact, moves final parameters by ~3e-2 in max-abs terms and swings final test accuracy by -0.020..+0.014 on this 3-epoch task (1 test sample = 0.002) — partial recovery silently changes results at well above tolerance, so 'resumed successfully' is an incomplete message unless the recovery tier is disclosed.

NOT established (no evidence claimed): anything GPU-related (cuDNN benchmark routing, TF32, nondeterministic atomic adds, cuBLAS workspace); multithreaded CPU reductions; DataLoader multi-worker RNG derivation and shuffling semantics; cross-Python/cross-library-version rng_state compatibility; and real data pipelines (augmentation, dropout, more RNG consumers than the four sources saved here). The defensible production claim remains: same hardware + same software stack => equivalence within tolerance; cross-hardware/cross-version => statistical equivalence only.
