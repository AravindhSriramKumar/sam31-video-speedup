"""Minimal test runner (no pytest dependency; the test_* functions also work under pytest).

Runs every `test_*` function of a module in definition order, prints PASS/FAIL, and writes the outcome plus
whatever each test returned (its measured numbers) to results/runs/eventful/<phase>/tests/<module>.json.
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eventful import env  # noqa: E402


def run(module_globals, name):
    phase = os.environ.get("EVENTFUL_PHASE", "2_correctness")
    only = os.environ.get("EVENTFUL_TESTS")            # optional comma-separated subset
    tests = [(n, f) for n, f in module_globals.items() if n.startswith("test_") and callable(f)]
    if only:
        tests = [(n, f) for n, f in tests if n in only.split(",")]
    results, failed = [], 0
    for n, f in tests:
        t0 = time.time()
        try:
            details = f()
            status = "PASS"
        except Exception as e:  # noqa: BLE001
            details, status = dict(error=repr(e), traceback=traceback.format_exc()), "FAIL"
            failed += 1
        dt = time.time() - t0
        print(f"[{status}] {n} ({dt:.1f}s)", flush=True)
        if status == "FAIL":
            print(details["traceback"], flush=True)
        results.append(dict(test=n, status=status, seconds=round(dt, 2), details=details))
    out = env.paths.RES_EVENTFUL / phase / "tests" / f"{name}.json"
    env.write_json(out, dict(module=name, passed=len(tests) - failed, failed=failed, results=results,
                             provenance=env.provenance()))
    print(f"{len(tests) - failed}/{len(tests)} passed -> {out}")
    sys.exit(1 if failed else 0)
