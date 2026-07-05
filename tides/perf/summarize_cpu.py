#!/usr/bin/env python3
"""Summarize WP2 CPU profiling JSON-lines logs into a compact table.

Reads all wp2_cpu_*.jsonl files in tides/perf/logs/, aggregates by operation +
params, and prints a table of cold/warm median times + accuracy. This is the
"diagnose and optimize" view: which operations are slow, how they scale, and
whether accuracy is within budget.
"""
import json
import sys
import glob
import os
from collections import defaultdict
from statistics import median

def main():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    files = sorted(glob.glob(os.path.join(log_dir, "wp2_cpu_*.jsonl")))
    if not files:
        print("No wp2_cpu_*.jsonl logs found in", log_dir, file=sys.stderr)
        return 1

    # Aggregate: key = (op, param_summary) -> {cold: [ms...], warm: [ms...]}
    agg = defaultdict(lambda: {"cold": [], "warm": []})
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                op = r["op"]
                params = r.get("params", {})
                # Compact param key (sort keys for stability)
                pkey = ",".join(f"{k}={v}" for k, v in sorted(params.items())
                              if k != "dummy")
                key = (op, pkey)
                agg[key][r["variant"]].append(r["ms"])

    print(f"{'operation':<22} {'params':<45} {'cold_ms':>10} {'warm_ms':>10} {'reps':>5}")
    print("-" * 95)
    for (op, pkey), times in sorted(agg.items()):
        cold = median(times["cold"]) if times["cold"] else float("nan")
        warm = median(times["warm"]) if times["warm"] else float("nan")
        reps = max(len(times["cold"]), len(times["warm"]))
        print(f"{op:<22} {pkey:<45} {cold:>10.3f} {warm:>10.3f} {reps:>5}")

    # Scaling analysis for radial_solve (n_r sweep)
    print("\n=== Scaling: radial_solve (selective tridiagonal eig) ===")
    print(f"{'n_r':>8} {'warm_ms':>10} {'ratio':>8}")
    scaling = []
    for (op, pkey), times in agg.items():
        if op != "radial_solve":
            continue
        n_r = None
        for kv in pkey.split(","):
            if kv.startswith("n_r="):
                n_r = int(kv.split("=")[1])
        if n_r is None:
            continue
        warm = median(times["warm"]) if times["warm"] else float("nan")
        scaling.append((n_r, warm))
    scaling.sort()
    prev = None
    for n_r, warm in scaling:
        ratio = f"{warm/prev:.2f}x" if prev else "-"
        print(f"{n_r:>8} {warm:>10.3f} {ratio:>8}")
        prev = warm

    return 0

if __name__ == "__main__":
    sys.exit(main())
