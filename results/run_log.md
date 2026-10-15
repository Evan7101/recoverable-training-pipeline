# Run log — recoverable-training experiment

Date: 2026-10-15. Machine: CPU-only Linux node, Python 3.13.9, stdlib only.
All commands run from `/hpc2hdd/home/ypu159/aw-tmp/recovery-baseline/`.

## 1. One uninterrupted run

Command:

```
python3 train_recoverable.py run --total 10
```

Output:

```
{"step": 10, "weight": 1.4562862250201976, "velocity": 0.3609445908320693, "total_steps": 10}
```

## 2. Manual interrupt/resume cycle using `run` + checkpoint

Commands:

```
python3 train_recoverable.py run --total 4 --save-checkpoint results/manual_ckpt.json
python3 train_recoverable.py run --total 10 --checkpoint results/manual_ckpt.json
```

Outputs:

```
{"step": 4, "weight": 5.427956709001653, "velocity": 0.8023047769317013, "total_steps": 4}
{"step": 10, "weight": 1.4562862250201976, "velocity": 0.3609445908320693, "total_steps": 4}
```

The resumed final `weight` (1.4562862250201976) and `velocity`
(0.3609445908320693) are identical to the uninterrupted run in step 1.
(`total_steps: 4` is checkpoint metadata recording when it was written; the
resume target comes from `--total 10` on the command line.)

## 3. Equivalence check (full output also in `results/equivalence.md`)

Command:

```
python3 train_recoverable.py compare --total 10 --splits 4 7
```

Key results (exit code 0):

- Split at step 4: BIT-EXACT (full state incl. rng_state): **True** — all 10
  steps match the reference bit-for-bit.
- Split at step 7: BIT-EXACT (full state incl. rng_state): **True** — all 10
  steps match the reference bit-for-bit.
- Partial recovery (no rng_state) split at 4: max abs per-step weight
  difference vs reference: **0.0240948**.
- Partial recovery (no rng_state) split at 7: max abs per-step weight
  difference vs reference: **0.00893984**.

Final reference trajectory (weight per step):
1.2014409032957793, 3.0038903909469843, 4.625651932069951, 5.427956709001653,
5.180124207477352, 4.09999880725795, 2.6845632407166304, 1.5395193093277626,
1.0953416341881284, 1.4562862250201976.

## 4. Negative tests (full capture in `results/negative_tests.txt`)

Command:

```
python3 train_recoverable.py selftest
```

Output (exit code 0):

```
PASS: missing file (exit=2, stderr=nonempty)
PASS: wrong format_version (exit=2, stderr=nonempty)
PASS: truncated/corrupt JSON (exit=2, stderr=nonempty)
PASS: missing required key (exit=2, stderr=nonempty)
```

Loader stderr messages (verbatim from the subprocess runs):

- missing file: `ERROR: checkpoint file not found: /tmp/negtest_v0xc8u0j/does_not_exist.json (refusing to start from scratch)`
- wrong format_version: `ERROR: unsupported format_version 999 in /tmp/negtest_v0xc8u0j/bad_version.json (this program understands version 1)`
- truncated/corrupt JSON: `ERROR: checkpoint /tmp/negtest_v0xc8u0j/truncated.json is corrupt/truncated JSON: Expecting value: line 318 column 6 (char 5634)`
- missing required key: `ERROR: checkpoint /tmp/negtest_v0xc8u0j/missing_key.json missing required key(s): velocity`

## Known limitations — what this experiment does NOT establish

- **Real frameworks**: this is hand-written pure-Python SGD+momentum; nothing
  here says anything about PyTorch/TensorFlow/JAX checkpoint formats or their
  resume semantics.
- **GPU / cuDNN nondeterminism**: CPU floats only. CUDA kernels (atomics,
  cuDNN benchmark mode, TF32) can make even uninterrupted runs nondeterministic,
  in which case checkpoint fidelity is beside the point.
- **Dataloader worker RNG and shuffling**: there is one global `random`
  stream consumed in a fixed order. Real pipelines have per-worker RNGs,
  shuffling epochs, and prefetching — all of which need their own save/restore.
- **Rich optimizer state**: only a single momentum scalar is saved. Adam/AdamW
  carry per-parameter first/second moments, step counters, and scheduler state,
  all of which must round-trip bit-exactly (not just approximately).
- **RNG reimplementation drift**: restoring `random.getstate()` is only
  bit-safe across the same Python version/stdlib; cross-version or
  cross-language resumes are not covered.
- **Distributed/multi-process training**: single process, single worker.
  Global steps in distributed setups require coordinated, consistent
  checkpointing across ranks.


---

## 修复：不一致 resume 目标与字段类型校验（2026-10-15）

代码审查发现两个违反"显式失败"原则的缺口，已修复（仅改动
`train_recoverable.py`；训练语义、RNG 流、checkpoint 格式均未变）：

1. **resume 目标步数不一致时静默成功**：`run_training` 中若 checkpoint
   的 `step` 已 >= 传入的 `--total`，循环空转并以 exit 0 退出。现在
   `total < checkpoint_step` 时显式报错退出 2；`total == checkpoint_step`
   视为幂等 no-op，允许成功（见下方验证）。
2. **字段类型错误抛裸 traceback**：`load_checkpoint` 在键检查后新增类型
   校验——`step`/`total_steps` 必须为非负 int（bool 不接受），
   `weight`/`velocity` 必须为 int/float（bool 不接受），否则走 `_load_fail`
   （stderr + exit 2）。
3. **selftest 增强**：新增负例 (e)（`step` 为字符串 `"abc"`），并对全部 5
   个负例额外断言 stderr 不含 `Traceback`。

### 验证命令与真实输出

```
$ python3 train_recoverable.py run --total 2 --checkpoint results/ckpt_split_4.json
ERROR: requested total steps 2 is behind checkpoint step 4 (refusing to go backwards)
(exit=2)
```

```
$ python3 train_recoverable.py run --total 10 --checkpoint results/bad_type_demo.json   # step="abc"
ERROR: checkpoint results/bad_type_demo.json field 'step' must be a non-negative int, got str: 'abc'
(exit=2)
```

幂等 no-op 仍成功：

```
$ python3 train_recoverable.py run --total 4 --checkpoint results/ckpt_split_4.json
{"step": 4, "weight": 5.427956709001653, "velocity": 0.8023047769317013, "total_steps": 4}
(exit=0)
```

修复后回归（均通过）：

```
$ python3 train_recoverable.py selftest
PASS: missing file (exit=2, stderr=nonempty, traceback=no)
PASS: wrong format_version (exit=2, stderr=nonempty, traceback=no)
PASS: truncated/corrupt JSON (exit=2, stderr=nonempty, traceback=no)
PASS: missing required key (exit=2, stderr=nonempty, traceback=no)
PASS: field 'step' has wrong type (str) (exit=2, stderr=nonempty, traceback=no)
(exit=0)
```

```
$ python3 train_recoverable.py compare --total 10 --splits 4 7
BIT-EXACT (full state incl. rng_state): True   # split 4
BIT-EXACT (full state incl. rng_state): True   # split 7
max abs per-step weight difference vs reference: 0.0240948   # partial, split 4
max abs per-step weight difference vs reference: 0.00893984  # partial, split 7
(exit=0)
```
