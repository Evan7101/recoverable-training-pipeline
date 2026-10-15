#!/usr/bin/env python3
"""Recoverable-training experiment: does an interrupted/resumed GD+momentum
run reproduce an uninterrupted run bit-for-bit?

Training model (toy but with the two things the baseline lacks):
  - optimizer state: velocity v, v = mu*v - lr*g ; w += v   (mu=0.9, lr=0.2)
  - RNG state: every step draws noise from the GLOBAL `random` module and adds
    it to the gradient, so the trajectory depends on the random stream.
Loss surface remains (w-3)^2; gradient g = 2*(w-3) + noise.

Checkpoint (JSON, atomic via tmp file + os.replace):
  format_version, step, weight, velocity, rng_state, total_steps

RNG state is random.getstate() -> (version, internal_tuple, gauss_next),
serialized as {"version": int, "internal": [ints], "gauss_next": float|null}
and restored faithfully via random.setstate(). No RNG draws may happen inside
save/load (that would desync the stream and is the classic recovery bug).

Usage:
  python3 train_recoverable.py run --total 10 [--checkpoint PATH]
  python3 train_recoverable.py compare --total 10 --splits 4 7
  python3 train_recoverable.py selftest
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

FORMAT_VERSION = 1
REQUIRED_KEYS = ("format_version", "step", "weight", "velocity", "rng_state", "total_steps")

LR = 0.2
MU = 0.9
NOISE_SIGMA = 0.05


def train_step(state):
    """One step of noisy gradient descent with momentum. Consumes exactly one
    RNG draw from the global random module."""
    noise = random.gauss(0.0, NOISE_SIGMA)
    g = 2.0 * (state["weight"] - 3.0) + noise
    state["velocity"] = MU * state["velocity"] - LR * g
    state["weight"] += state["velocity"]
    state["step"] += 1
    return state


def initial_state(total_steps):
    return {"step": 0, "weight": 0.0, "velocity": 0.0, "total_steps": total_steps}


def rng_state_to_json():
    version, internal, gauss_next = random.getstate()
    return {
        "version": version,
        "internal": list(internal),
        "gauss_next": gauss_next,
    }


def rng_state_from_json(d):
    return (int(d["version"]), tuple(int(x) for x in d["internal"]), d["gauss_next"])


def build_checkpoint(state):
    return {
        "format_version": FORMAT_VERSION,
        "step": state["step"],
        "weight": state["weight"],
        "velocity": state["velocity"],
        "rng_state": rng_state_to_json(),
        "total_steps": state["total_steps"],
    }


def save_checkpoint(state, path):
    ckpt = build_checkpoint(state)
    path = os.fspath(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ckpt_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ckpt, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ckpt


def _load_fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def load_checkpoint(path):
    """Load checkpoint or fail loudly (stderr + exit 2). Never silently
    reinitialize."""
    if not os.path.exists(path):
        _load_fail(f"checkpoint file not found: {path} "
                    f"(refusing to start from scratch)")
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError as e:
        _load_fail(f"cannot read checkpoint {path}: {e}")
    try:
        ckpt = json.loads(text)
    except json.JSONDecodeError as e:
        _load_fail(f"checkpoint {path} is corrupt/truncated JSON: {e}")
    if not isinstance(ckpt, dict):
        _load_fail(f"checkpoint {path} is not a JSON object")
    if ckpt.get("format_version") != FORMAT_VERSION:
        _load_fail(f"unsupported format_version "
                   f"{ckpt.get('format_version')!r} in {path} "
                   f"(this program understands version {FORMAT_VERSION})")
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        _load_fail(f"checkpoint {path} missing required key(s): "
                   f"{', '.join(missing)}")
    for key in ("step", "total_steps"):
        v = ckpt[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            _load_fail(f"checkpoint {path} field '{key}' must be a "
                       f"non-negative int, got {type(v).__name__}: {v!r}")
    for key in ("weight", "velocity"):
        v = ckpt[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _load_fail(f"checkpoint {path} field '{key}' must be a number "
                       f"(int or float), got {type(v).__name__}: {v!r}")
    try:
        random.setstate(rng_state_from_json(ckpt["rng_state"]))
    except (KeyError, TypeError, ValueError) as e:
        _load_fail(f"checkpoint {path} has malformed rng_state: {e}")
    return {
        "step": ckpt["step"],
        "weight": ckpt["weight"],
        "velocity": ckpt["velocity"],
        "total_steps": ckpt["total_steps"],
    }


def run_training(total, checkpoint_path=None, restore_rng=True,
                 start_state=None):
    """Run to `total` steps. If checkpoint_path is given, resume from it
    (restoring RNG unless restore_rng=False). Returns (final_state,
    trajectory) where trajectory[step] is the weight AFTER that step
    (1-indexed)."""
    if start_state is not None:
        state = dict(start_state)
    elif checkpoint_path is not None:
        state = load_checkpoint(checkpoint_path)
        if state["step"] > total:
            print(f"ERROR: requested total steps {total} is behind "
                  f"checkpoint step {state['step']} "
                  f"(refusing to go backwards)", file=sys.stderr)
            sys.exit(2)
        if not restore_rng:
            # Partial recovery: deliberately do NOT restore the RNG stream.
            # Re-seed fresh so the divergence is attributable to the lost
            # rng_state alone.
            random.seed(12345)
    else:
        state = initial_state(total)
        random.seed(42)

    trajectory = {0: state["weight"]}
    while state["step"] < total:
        train_step(state)
        trajectory[state["step"]] = state["weight"]
    return state, trajectory


def cmd_run(args):
    state, _ = run_training(args.total, args.checkpoint)
    if args.save_checkpoint:
        save_checkpoint(state, args.save_checkpoint)
    out = {
        "step": state["step"],
        "weight": state["weight"],
        "velocity": state["velocity"],
        "total_steps": state["total_steps"],
    }
    print(json.dumps(out))


def cmd_compare(args):
    total = args.total
    splits = sorted(args.splits)
    os.makedirs("results", exist_ok=True)

    random.seed(42)
    ref_state, ref_traj = run_training(total)
    ref_weights = [ref_traj[s] for s in range(1, total + 1)]

    lines = []
    out = lines.append
    all_bit_exact = True
    out(f"# Equivalence: interrupted/resumed vs uninterrupted (total={total})")
    out("")
    out(f"Reference: GD+momentum (mu={MU}, lr={LR}) with per-step Gaussian "
        f"noise (sigma={NOISE_SIGMA}) from the global `random` module.")
    out("")

    # --- interrupted/resumed per split point ---
    for n in splits:
        if not (0 < n < total):
            print(f"split {n} must be in (0, {total})", file=sys.stderr)
            sys.exit(2)
        ckpt_path = os.path.join("results", f"ckpt_split_{n}.json")
        random.seed(42)
        part_state, part_traj = run_training(n)
        save_checkpoint(part_state, ckpt_path)

        # resume in-process, but through the real JSON checkpoint round-trip
        random.seed(999)  # clobber global RNG to prove restore works
        res_state, res_traj = run_training(total, ckpt_path)
        # pre-split steps come from the identical interrupted segment
        res_traj = {**part_traj, **res_traj}

        matches = [res_traj[s] == ref_traj[s] for s in range(1, total + 1)]
        bit_exact = all(matches)
        all_bit_exact = all_bit_exact and bit_exact

        out(f"## Split at step {n}")
        out("")
        out("| step | reference weight | resumed weight | bit-exact |")
        out("|-----:|-----------------:|---------------:|:---------:|")
        for s in range(1, total + 1):
            m = "yes" if matches[s - 1] else "NO"
            out(f"| {s} | {ref_traj[s]!r} | {res_traj[s]!r} | {m} |")
        out("")
        out(f"BIT-EXACT (full state incl. rng_state): {bit_exact}")
        out("")

    # --- partial recovery: no rng_state restore ---
    partial_results = []
    for n in splits:
        ckpt_path = os.path.join("results", f"ckpt_split_{n}.json")
        random.seed(999)
        res_state, res_traj = run_training(total, ckpt_path,
                                           restore_rng=False)
        res_traj = {s: ref_traj[s] for s in range(1, n + 1)} | res_traj
        diffs = [abs(res_traj[s] - ref_traj[s]) for s in range(1, total + 1)]
        max_diff = max(diffs)
        partial_results.append((n, max_diff))
        out(f"## Partial recovery (no rng_state) split at step {n}")
        out("")
        out(f"max abs per-step weight difference vs reference: {max_diff:.6g}")
        out("")

    text = "\n".join(lines) + "\n"
    with open(os.path.join("results", "equivalence.md"), "w") as f:
        f.write(text)
    print(text, end="")

    if not all_bit_exact:
        print("FAIL: at least one split is not bit-exact", file=sys.stderr)
        sys.exit(1)


def cmd_selftest(args):
    os.makedirs("results", exist_ok=True)
    script = os.path.abspath(__file__)
    workdir = tempfile.mkdtemp(prefix="negtest_")

    # a valid checkpoint to mutate
    random.seed(42)
    state, _ = run_training(4, start_state=initial_state(10))
    good_path = os.path.join(workdir, "good.json")
    save_checkpoint(state, good_path)
    with open(good_path) as f:
        good = json.load(f)

    cases = []

    # (a) missing file
    cases.append(("missing file",
                  os.path.join(workdir, "does_not_exist.json"), None))

    # (b) wrong format_version
    bad = dict(good, format_version=999)
    p = os.path.join(workdir, "bad_version.json")
    with open(p, "w") as f:
        json.dump(bad, f)
    cases.append(("wrong format_version", p, None))

    # (c) truncated/corrupt JSON
    p = os.path.join(workdir, "truncated.json")
    with open(good_path) as f:
        full = f.read()
    with open(p, "w") as f:
        f.write(full[: len(full) // 2])
    cases.append(("truncated/corrupt JSON", p, None))

    # (d) missing required key
    bad = {k: v for k, v in good.items() if k != "velocity"}
    p = os.path.join(workdir, "missing_key.json")
    with open(p, "w") as f:
        json.dump(bad, f)
    cases.append(("missing required key", p, None))

    # (e) field with wrong type (step is a string)
    bad = dict(good, step="abc")
    p = os.path.join(workdir, "bad_field_type.json")
    with open(p, "w") as f:
        json.dump(bad, f)
    cases.append(("field 'step' has wrong type (str)", p, None))

    report = []
    all_ok = True
    for name, path, _ in cases:
        proc = subprocess.run(
            [sys.executable, script, "run", "--total", "10",
             "--checkpoint", path],
            capture_output=True, text=True,
        )
        no_traceback = "Traceback" not in proc.stderr
        ok = (proc.returncode != 0 and proc.stderr.strip() != ""
              and no_traceback)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        report.append(f"=== {name} ===")
        report.append(f"status: {status}")
        report.append(f"exit code: {proc.returncode}")
        report.append(f"stderr: {proc.stderr.strip()}")
        report.append(f"stdout: {proc.stdout.strip()}")
        report.append("")
        print(f"{status}: {name} (exit={proc.returncode}, "
              f"stderr={'nonempty' if proc.stderr.strip() else 'EMPTY'}, "
              f"traceback={'no' if no_traceback else 'YES'})")

    with open(os.path.join("results", "negative_tests.txt"), "w") as f:
        f.write("\n".join(report))
    sys.exit(0 if all_ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--total", type=int, required=True)
    pr.add_argument("--checkpoint", default=None)
    pr.add_argument("--save-checkpoint", default=None)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare")
    pc.add_argument("--total", type=int, required=True)
    pc.add_argument("--splits", type=int, nargs="+", required=True)
    pc.set_defaults(func=cmd_compare)

    ps = sub.add_parser("selftest")
    ps.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
