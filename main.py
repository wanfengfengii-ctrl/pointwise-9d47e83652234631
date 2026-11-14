#!/usr/bin/env python3
"""Command line interface for the dynamic triangle counter.

Three sub-commands are exposed (see ``.pointwise-runtime.json``):

``verify``
    Business checks: the worked example, dedupe semantics and differential
    runs of the incremental counter against the independent enumerator.

``demo``
    Prints the worked example step by step, including a delete.

``help``
    Shows usage.  This is the default when no sub-command is given.

Every command prints human readable text to stdout and exits non-zero on
failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

from reference import count_triples, run_differential, hub_scenario


def _now() -> float:
    return time.perf_counter()


def cmd_demo(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="main.py demo")
    parser.add_argument(
        "--verbose", action="store_true", help="print raw return values too"
    )
    args = parser.parse_args(argv)

    from triangle import Counter

    counter = Counter(8)
    steps = [
        ("insert", "R", 1, 2),
        ("insert", "S", 2, 3),
        ("insert", "T", 3, 1),
    ]
    print("R(1,2), S(2,3), T(3,1)")
    for op, table, x, y in steps:
        changed = counter.insert(table, x, y)
        print(
            "  insert %s(%d,%d) -> changed=%s count=%d"
            % (table, x, y, changed, counter.count())
        )
    print("  count() = %d" % counter.count())
    if args.verbose:
        counter.validate()

    changed = counter.delete("S", 2, 3)
    print("  delete S(2,3) -> changed=%s count=%d" % (changed, counter.count()))
    if args.verbose:
        counter.validate()

    changed = counter.insert("S", 2, 3)
    print("  reinsert S(2,3) -> changed=%s count=%d" % (changed, counter.count()))
    if args.verbose:
        counter.validate()
    return 0


def cmd_verify(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="main.py verify")
    parser.add_argument(
        "--seed", type=int, default=20261114, help="base random seed"
    )
    parser.add_argument(
        "--steps", type=int, default=400, help="operations per differential run"
    )
    parser.add_argument(
        "--runs", type=int, default=12, help="number of differential runs"
    )
    parser.add_argument(
        "--quick", action="store_true", help="skip the (slower) hub scenario"
    )
    args = parser.parse_args(argv)

    from triangle import Counter

    failures = 0

    # 1. Worked example from the statement, with return values.
    c = Counter(8)
    assert c.insert("R", 1, 2) is True
    assert c.insert("S", 2, 3) is True
    assert c.insert("T", 3, 1) is True
    assert c.count() == 1
    assert c.delete("S", 2, 3) is True
    assert c.count() == 0
    # duplicate insert / delete of missing record do not change anything
    assert c.insert("R", 1, 2) is False
    assert c.delete("S", 9, 9) is False
    assert c.count() == 0
    # restoring the record restores the triangle
    assert c.insert("S", 2, 3) is True
    assert c.count() == 1
    c.validate()
    print("example: R(1,2), S(2,3), T(3,1) -> 1; delete S(2,3) -> 0; restore -> 1")

    # 2. Partition tagging: identical integers on different sides are distinct.
    c = Counter(8)
    c.insert("R", 0, 0)
    c.insert("S", 0, 0)
    c.insert("T", 0, 0)
    assert c.count() == 1, "same integer in all three tables is a triangle"
    c.delete("T", 0, 0)
    assert c.count() == 0
    c.validate()
    print("partition keys: equal integers in R/S/T form exactly one triple")

    # 3. Independent enumeration vs incremental counter.
    for run in range(args.runs):
        seed = args.seed + run
        B = 25 + (run * 7) % 60
        n_keys = 5 + run % 9
        steps = args.steps
        try:
            run_differential(
                B,
                seed,
                steps,
                n_keys=n_keys,
                validate_every=7,
                brute_every=3,
                key_offset=-(run % 3),
            )
        except AssertionError as exc:
            failures += 1
            print("differential FAILED (seed %d, B %d): %s" % (seed, B, exc))
    if failures:
        print("verify: %d differential run(s) failed" % failures)
        return 1
    print(
        "differential: %d runs x %d ops vs independent enumeration, all equal"
        % (args.runs, args.steps)
    )

    # 4. Heavy-light promotion / memoised heavy pairs / demotion.
    if not args.quick:
        try:
            hub_scenario(args.seed, B=20000)
        except AssertionError as exc:
            print("hub scenario FAILED: %s" % exc)
            return 1
        print("hub scenario: heavy vertices promoted, validated, counts equal")

    print("verify: all checks passed")
    return 0


def cmd_bench(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="main.py bench")
    parser.add_argument("--B", type=int, default=200000)
    parser.add_argument("--ops", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workload",
        default="all",
        choices=["all", "hotspot", "light", "count", "scaling"],
    )
    args = parser.parse_args(argv)

    import subprocess

    # One fresh process per workload keeps the reported peak RSS meaningful.
    parts = (
        ["hotspot", "light", "count", "scaling"]
        if args.workload == "all"
        else [args.workload]
    )
    bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
    for part in parts:
        subprocess.run(
            [
                sys.executable,
                "-I",
                bench_path,
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Dynamic R-S-T triangle counter (stdlib only).",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("verify", help="run business checks")
    sub.add_parser("demo", help="show insert/delete behaviour")
    sub.add_parser("bench", help="measure access counts, time and memory")
    return parser


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in ("help", "-h", "--help"):
        parser.print_help()
        return 0
    args, rest = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handlers = {"verify": cmd_verify, "demo": cmd_demo, "bench": cmd_bench}
    return handlers[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
