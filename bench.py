"""Performance measurement for the dynamic triangle counter.

Two workloads are measured, because they stress opposite ends of the
heavy/light design:

``hotspot``
    One hub vertex per partition, with degree far above 2*sqrt(B), plus
    churn among ordinary vertices.  Intersections touching a heavy vertex
    come from the memoised heavy-pair table, so updates are cheap.

``light``
    No-triangle graph where every degree is just below 2*sqrt(B).  Each
    insert of an R edge must intersect two adjacency sets of size ~sqrt(B);
    this is the worst case of the amortised bound.

Each workload is run twice from an identical build: once with
``track_accesses=True`` (gives the exact number of adjacency / heavy-table
slots read or written, the implementation's unit of work) and once in
production mode (gives honest wall time; the tracked run uses a Python loop
to count, production uses C-level set intersection).  Both runs must end at
the same ``count()`` and the production run is validated.

Run ``python3 main.py bench`` (or ``python3 bench.py``) for everything.
"""

from __future__ import annotations

import gc
import math
import os
import random
import resource
import sys
import time
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triangle import Counter

Plan = List[Tuple[str, int, int]]


def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _apply_plan(
    counter: Counter, plan: Plan, ops: int
) -> Tuple[int, float, int]:
    """Toggle the first ``ops`` records of ``plan``.

    Returns ``(changes, seconds, max accesses in a single op)``.  The last
    value is only meaningful with ``track_accesses=True``.
    """
    present: set = set()
    changed = 0
    max_op = 0
    t0 = time.perf_counter()
    for i in range(ops):
        key = plan[i % len(plan)]
        table, x, y = key
        before = counter.accesses
        if key in present:
            present.discard(key)
            changed += counter.delete(table, x, y)
        else:
            present.add(key)
            changed += counter.insert(table, x, y)
        max_op = max(max_op, counter.accesses - before)
    return changed, time.perf_counter() - t0, max_op


# ----------------------------------------------------------------------
# Workload builders
# ----------------------------------------------------------------------
def build_hotspot(B: int, seed: int) -> Tuple[Callable[[Counter], None], Plan]:
    rng = random.Random(seed)
    k = max(10, min(B // 48, 20000))

    def build(counter: Counter) -> None:
        for i in range(1, k + 1):
            counter.insert("R", 0, i)  # A-hub -- B_i
            counter.insert("T", i, 0)  # C_i -- A-hub
            counter.insert("S", i, 0)  # B-hub -- C_i
            counter.insert("R", i, 0)  # B-hub -- A_i
            counter.insert("S", 0, i)  # C-hub -- B_i
            counter.insert("T", 0, i)  # C-hub -- A_i
        counter.insert("T", 0, 0)  # closes A-hub--B_i--C-hub triangles

    plan: Plan = []
    for i in range(k + 1, 2 * k + 1):
        plan.append(("R", 0, i))  # hub <-> light (memoised side)
        plan.append(("T", i, 0))
        plan.append(("S", i, 0))
        plan.append(("R", i, i + 1))  # light <-- light fallback
    rng.shuffle(plan)
    return build, plan


def build_light(B: int, seed: int) -> Tuple[Callable[[Counter], None], Plan]:
    D = max(1, math.isqrt(B) + (0 if math.isqrt(B) ** 2 == B else 1))
    P = 2 * D
    t = P - 1  # keep every vertex below the promotion threshold
    half = t // 2
    m = max(2, B // (2 * t))

    def build(counter: Counter) -> None:
        for i in range(m):
            for j in range(half):
                counter.insert("T", i * 1000 + j, i)  # C block -> A_i
            for j in range(half, t):
                counter.insert("S", i, i * 1000 + j)  # B_i -> C block
            counter.insert("R", i, (i + 1) % m)  # no triangle possible

    plan: Plan = [("R", i, j) for i in range(m) for j in range(m)]
    random.Random(seed).shuffle(plan)
    return build, plan


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def measure(B: int, ops: int, seed: int, kind: str) -> Dict[str, float]:
    builder = build_hotspot if kind == "hotspot" else build_light
    build, plan = builder(B, seed)
    ops = max(1, min(ops, len(plan) * 4))

    tracked = Counter(B, track_accesses=True)
    build(tracked)
    build_accesses = tracked.accesses
    build_records = tracked._m
    build_heavy = len(tracked._heavy)
    ops_used = min(ops, len(plan) * 2)
    start = tracked.accesses
    changed, _, max_op = _apply_plan(tracked, plan, ops_used)
    accesses = tracked.accesses - start
    tracked_count = tracked.count()
    tracked.validate()
    tracked_entries = (
        sum(len(s) for s in tracked._adj.values())
        + sum(len(s) for s in tracked._heavy_nbrs.values())
        + sum(len(r) for r in tracked._hc.values())
    )
    del tracked
    gc.collect()

    # Timing/memory from a clean, non-instrumented instance.
    plain = Counter(B)
    build(plain)
    changed2, seconds, _ = _apply_plan(plain, plan, ops_used)
    plain_count = plain.count()
    plain.validate()
    if changed != changed2 or tracked_count != plain_count:
        raise AssertionError("timed and tracked runs diverged")
    state_entries = (
        sum(len(s) for s in plain._adj.values())
        + sum(len(s) for s in plain._heavy_nbrs.values())
        + sum(len(r) for r in plain._hc.values())
    )
    if state_entries != tracked_entries:
        raise AssertionError("state layout differs between runs")

    return {
        "label": kind,
        "B": B,
        "ops": ops_used,
        "changed": changed,
        "accesses": accesses,
        "accesses_per_op": accesses / ops_used,
        "max_op_accesses": max_op,
        "us_per_op": seconds / ops_used * 1e6,
        "records": plain._m,
        "build_records": build_records,
        "heavy": build_heavy,
        "build_accesses": build_accesses,
        "triangles": plain_count,
        "state_entries": state_entries,
        "peak_kb": _rss_kb(),
    }


def scaling_table(sizes: List[int], seed: int = 3) -> List[Dict[str, float]]:
    return [measure(B, ops=min(2000, B // 8), seed=seed, kind="light") for B in sizes]


def _print_stats(s: Dict[str, float]) -> None:
    print(
        "  [%s] B=%d: build %d records -> %d live (%d heavy), %d build accesses"
        % (
            s["label"],
            s["B"],
            s["build_records"],
            s["records"],
            s["heavy"],
            s["build_accesses"],
        )
    )
    print(
        "    churn: %d ops (%d changed record set), %d accesses, %.2f us/op"
        % (s["ops"], s["changed"], s["accesses"], s["us_per_op"])
    )
    print(
        "    accesses/op: %.1f (worst single op %d)   sqrt(B)=%.0f   "
        "count()=%d   state entries=%d   peak RSS %.1f MiB"
        % (
            s["accesses_per_op"],
            s["max_op_accesses"],
            math.sqrt(s["B"]),
            s["triangles"],
            s["state_entries"],
            s["peak_kb"] / 1024,
        )
    )


def run_bench(
    B: int = 200000,
    ops: int = 20000,
    seed: int = 1,
    workload: str = "all",
) -> None:
    gc.collect()
    if workload in ("hotspot", "all"):
        print("== hotspot workload (hubs stay heavy; memoised intersections) ==")
        _print_stats(measure(B, ops, seed, "hotspot"))

    if workload in ("light", "all"):
        print("== light workload (all degrees just under 2*sqrt(B)) ==")
        _print_stats(measure(B, ops, seed, "light"))

    if workload in ("count", "all"):
        counter = Counter(10, track_accesses=True)
        counter.insert("R", 1, 2)
        counter.insert("S", 2, 3)
        counter.insert("T", 3, 1)
        before = counter.accesses
        n = 200000
        t0 = time.perf_counter()
        s = 0
        for _ in range(n):
            s += counter.count()
        dt = time.perf_counter() - t0
        print("== count() ==")
        print(
            "  %d calls in %.3f s (%.3f us/call), %d state accesses, checksum %d"
            % (n, dt, dt / n * 1e6, counter.accesses - before, s)
        )

    if workload in ("scaling", "all"):
        sizes = [10_000, 40_000, 160_000, 640_000]
        print("== scaling of the light workload (accesses per op) ==")
        print(
            "  %10s %10s %12s %10s %10s %10s"
            % ("B", "sqrt(B)", "accesses/op", "us/op", "records", "MiB")
        )
        for row in scaling_table(sizes, seed):
            print(
                "  %10d %10.0f %12.1f %10.2f %10d %10.1f"
                % (
                    row["B"],
                    math.sqrt(row["B"]),
                    row["accesses_per_op"],
                    row["us_per_op"],
                    row["records"],
                    row["peak_kb"] / 1024,
                )
            )
        print(
            "  memory is O(B): one adjacency entry per live record plus a\n"
            "  small heavy-pair table.  Each row is a fresh measurement in a\n"
            "  separate process, so RSS is not cumulative across rows."
        )


def _child(workload: str, B: int, ops: int, seed: int) -> None:
    """Run one workload in this (fresh) process; used for clean RSS numbers."""
    run_bench(B=B, ops=ops, seed=seed, workload=workload)


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(prog="bench.py")
    parser.add_argument("--B", type=int, default=200000)
    parser.add_argument("--ops", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workload",
        default="all",
        choices=["all", "hotspot", "light", "count", "scaling"],
    )
    args = parser.parse_args()
    if args.workload == "all":
        # Fresh process per workload so peak RSS is not cumulative.
        for part in ("hotspot", "light", "count", "scaling"):
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    __file__,
                    "--workload",
                    part,
                    "--B",
                    str(args.B),
                    "--ops",
                    str(args.ops),
                    "--seed",
                    str(args.seed),
                ],
                check=True,
            )
    else:
        _child(args.workload, args.B, args.ops, args.seed)

