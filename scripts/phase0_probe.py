#!/usr/bin/env python3
"""
Phase 0 probe for the AI-in-the-loop discovery platform.

Reproducibly checks whether the discovery loop can run against `habitat_pipeline`,
and surfaces exactly where it breaks. Runs four checks:

  1. ENV      - Python + required-dependency probe.
  2. TESTS    - core mock-data analysis tests (no SMB share needed).
  3. REAL     - a real-session decode CLI (needs the //nearline share mounted).
  4. LOOP     - synthetic Runner -> Interpreter demo, incl. the multiple-
                comparison rigor guardrail, using known ground truth.

Designed to run BOTH in a data-less sandbox (REAL fails gracefully) and on a
Janelia workstation where the share is mounted (REAL should succeed).

Usage
-----
    python scripts/phase0_probe.py                      # all checks, defaults
    python scripts/phase0_probe.py --animal-id 631 --session-id 20251216
    python scripts/phase0_probe.py --skip-real          # data-less machines
    python scripts/phase0_probe.py --only loop          # just the loop demo

Exit code is 0 if every non-skipped, non-REAL check passes. REAL is reported
but never fails the run (so the script is useful on data-less machines too).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Core analysis tests that rely only on mock data (no GUI / streamlit, no share).
CORE_TEST_FILES = [
    "test_social_spatial_fields.py",
    "test_inter_brain_dynamics.py",
    "test_multi_animal_session.py",
    "test_kilosort_data_analysis.py",
    "test_decode_partner_distance.py",
]

REQUIRED_DEPS = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "h5py"]


# --------------------------------------------------------------------------- #
# small reporting helpers
# --------------------------------------------------------------------------- #
class Reporter:
    def __init__(self) -> None:
        self.results: dict[str, str] = {}  # name -> PASS / FAIL / SKIP / INFO

    def header(self, title: str) -> None:
        print("\n" + "=" * 68)
        print(title)
        print("=" * 68)

    def record(self, name: str, status: str, note: str = "") -> None:
        self.results[name] = status
        tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]",
               "INFO": "[INFO]"}.get(status, status)
        print(f"  {tag} {name}" + (f" - {note}" if note else ""))

    def summary(self) -> int:
        self.header("PHASE 0 SUMMARY")
        for name, status in self.results.items():
            print(f"  {status:5s}  {name}")
        # REAL and SKIP/INFO never fail the run; only hard checks do.
        hard_fail = any(
            s == "FAIL" and n != "REAL: real-session decode CLI"
            for n, s in self.results.items()
        )
        print("\nResult:", "ISSUES FOUND" if hard_fail else "OK "
              "(REAL leg is informational; see notes above)")
        return 1 if hard_fail else 0


# --------------------------------------------------------------------------- #
# 1. ENV
# --------------------------------------------------------------------------- #
def check_env(rep: Reporter) -> None:
    rep.header("1. ENVIRONMENT")
    print(f"  python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  repo    : {REPO_ROOT}")
    missing = []
    for dep in REQUIRED_DEPS:
        try:
            importlib.import_module(dep)
        except Exception:  # noqa: BLE001
            missing.append(dep)
    if missing:
        rep.record("ENV: required deps", "FAIL",
                   f"missing {missing} -> pip install -r requirements.txt")
    else:
        rep.record("ENV: required deps", "PASS", "all core deps importable")


# --------------------------------------------------------------------------- #
# 2. TESTS (mock data)
# --------------------------------------------------------------------------- #
def check_tests(rep: Reporter, full: bool) -> None:
    rep.header("2. MOCK-DATA TESTS (no share needed)")
    tests_dir = REPO_ROOT / "tests"
    if full:
        target = ["tests"]
        print("  running FULL suite (may need GUI deps like streamlit)")
    else:
        target = [str(tests_dir / f) for f in CORE_TEST_FILES
                  if (tests_dir / f).exists()]
        print(f"  running {len(target)} core analysis test files")
    cmd = [sys.executable, "-m", "pytest", "-q", *target]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
    print("  " + tail[0])
    if proc.returncode == 0:
        rep.record("TESTS: core mock suite", "PASS", f"{dt:.1f}s")
    else:
        rep.record("TESTS: core mock suite", "FAIL",
                   "see pytest output above")
        print(proc.stdout[-1500:])


# --------------------------------------------------------------------------- #
# 3. REAL session decode CLI (needs the share)
# --------------------------------------------------------------------------- #
def check_real(rep: Reporter, animal_id: str, session_id: str,
               config: str | None) -> None:
    rep.header("3. REAL-SESSION DECODE CLI (needs //nearline share)")
    cmd = [sys.executable, "-m", "ephys.decode_opponent_identity",
           "--animal_id", animal_id, "--session_id", session_id,
           "--use_quality_cells"]
    if config:
        cmd += ["--config_path", config]
    print("  $ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        rep.record("REAL: real-session decode CLI", "INFO",
                   "timed out after 30 min")
        return
    out = (proc.stdout + proc.stderr).strip()
    print("  " + "\n  ".join(out.splitlines()[-8:]))
    lowered = out.lower()
    if proc.returncode == 0 and "error loading data" not in lowered:
        rep.record("REAL: real-session decode CLI", "PASS",
                   "real result produced -> ready for Phase 1")
    elif "no session directory" in lowered or "error loading data" in lowered:
        rep.record("REAL: real-session decode CLI", "INFO",
                   "data not reachable here (expected off-workstation)")
    else:
        rep.record("REAL: real-session decode CLI", "FAIL",
                   "unexpected error - inspect output above")


# --------------------------------------------------------------------------- #
# 4. LOOP demo (synthetic, self-contained) + rigor guardrail
# --------------------------------------------------------------------------- #
def check_loop(rep: Reporter, seed: int) -> None:
    rep.header("4. LOOP MECHANICS DEMO (synthetic ground truth)")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import numpy as np
        from ephys._lda_decoding import run_population_per_cell_decode
    except Exception as exc:  # noqa: BLE001
        rep.record("LOOP: runner+interpreter", "FAIL", f"import failed: {exc}")
        return

    rng = np.random.default_rng(seed)
    dur, n_cells, n_tuned = 200.0, 24, 6
    event_times = np.sort(rng.uniform(5, dur - 5, 60))
    labels = rng.integers(0, 2, size=len(event_times))  # opponent A(0)/B(1)

    spike_times_list, cluster_ids = [], []
    for c in range(n_cells):
        base = rng.uniform(2, 8)  # Hz
        spikes = np.sort(rng.uniform(0, dur, int(base * dur)))
        if c < n_tuned:  # tuned cells burst after their preferred opponent
            pref = c % 2
            for et, lab in zip(event_times, labels):
                if lab == pref:
                    spikes = np.concatenate(
                        [spikes, rng.uniform(et, et + 1.0, rng.poisson(6))])
        spike_times_list.append(np.sort(spikes))
        cluster_ids.append(c)

    # --- RUNNER leg: real analysis call -> real result-dict ---
    cell_results, ok_ids, accs = run_population_per_cell_decode(
        spike_times_list, cluster_ids, event_times, labels,
        time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
        min_events_per_class=5, progress_every=1000)
    accs = np.asarray(accs)

    # --- INTERPRETER leg: read schema, summarize ---
    chance = float(max(np.mean(labels == 0), np.mean(labels == 1)))
    best_id = ok_ids[int(np.argmax(accs))]
    ex = cell_results[best_id]
    print("  runner result-dict (best cell):",
          {k: ex[k] for k in ("status", "accuracy", "n_events") if k in ex})
    print(f"  cells decoded : {len(accs)}/{n_cells}")
    print(f"  chance level  : {chance:.2f}")
    print(f"  mean CV acc   : {accs.mean():.3f}")

    # --- RIGOR guardrail: naive screen vs multiple-comparison control ---
    naive_hits = int(np.sum(accs > chance + 0.05))
    print(f"\n  naive screen (acc > chance+0.05): {naive_hits} 'significant' "
          f"cells vs {n_tuned} truly tuned")
    try:
        # Permutation-free illustrative correction: treat accuracy as a z-like
        # score and apply Benjamini-Hochberg over per-cell p's from a binomial
        # approximation. (Real pipeline should use label-permutation nulls.)
        from math import comb
        n_ev = len(labels)
        k = np.clip((accs * n_ev).round().astype(int), 0, n_ev)
        p = np.array([sum(comb(n_ev, j) for j in range(ki, n_ev + 1))
                      / 2 ** n_ev for ki in k])
        from statsmodels.stats.multitest import fdrcorrection
        rej, _ = fdrcorrection(p, alpha=0.05)
        print(f"  FDR-corrected (BH, alpha=0.05)   : {int(rej.sum())} "
              "significant cells")
        note = (f"naive {naive_hits} -> FDR {int(rej.sum())} "
                f"(truth {n_tuned}); correction matters")
    except Exception:  # noqa: BLE001  (statsmodels optional)
        print("  [statsmodels not installed -> skipped FDR illustration]")
        note = (f"naive screen flagged {naive_hits} vs {n_tuned} true; "
                "apply FDR/Bonferroni in the loop")

    rep.record("LOOP: runner+interpreter", "PASS", "result-dict produced")
    rep.record("LOOP: rigor guardrail", "INFO", note)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 discovery-loop probe.")
    ap.add_argument("--animal-id", default="631")
    ap.add_argument("--session-id", default="20251216")
    ap.add_argument("--config", default=None,
                    help="path to a DataStorageManager config JSON")
    ap.add_argument("--only", choices=["env", "tests", "real", "loop"],
                    help="run only one check")
    ap.add_argument("--skip-real", action="store_true",
                    help="skip the real-session CLI (for data-less machines)")
    ap.add_argument("--full-tests", action="store_true",
                    help="run the whole test suite (may need GUI deps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep = Reporter()
    print("Phase 0 probe -", time.strftime("%Y-%m-%d %H:%M:%S"))

    run = (lambda name: args.only is None or args.only == name)
    if run("env"):
        check_env(rep)
    if run("tests"):
        check_tests(rep, full=args.full_tests)
    if run("real"):
        if args.skip_real:
            rep.record("REAL: real-session decode CLI", "SKIP", "--skip-real")
        else:
            check_real(rep, args.animal_id, args.session_id, args.config)
    if run("loop"):
        check_loop(rep, args.seed)

    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
