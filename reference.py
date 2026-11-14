"""Independent reference implementation used by tests and ``main.py verify``.

It deliberately knows nothing about :class:`triangle.Counter`: relations are
plain sets of pairs and triangles are enumerated directly.
"""

from __future__ import annotations

import random
from typing import Iterable, Set, Tuple

Pair = Tuple[int, int]
Record = Tuple[str, int, int]


def count_triples(
    R: Iterable[Pair], S: Iterable[Pair], T: Iterable[Pair]
) -> int:
    """Count (a, b, c) with R(a,b), S(b,c) and T(c,a), by direct enumeration."""
    by_b: dict[int, list[int]] = {}
    for b, c in S:
        by_b.setdefault(b, []).append(c)
    total = 0
    for a, b in R:
        for c in by_b.get(b, ()):
            if (c, a) in T:
                total += 1
    return total


def run_differential(
    B: int,
    seed: int,
    steps: int,
    n_keys: int = 12,
    *,
    validate_every: int = 0,
    brute_every: int = 1,
    key_offset: int = 0,
) -> int:
    """Drive a Counter and a plain-set model in lockstep.

    Returns the number of operations checked.  Raises ``AssertionError`` with
    a diagnostic on the first discrepancy.
    """
    from triangle import Counter

    rng = random.Random(seed)
    counter = Counter(B)
    sets: dict[str, Set[Pair]] = {"R": set(), "S": set(), "T": set()}

    for step in range(steps):
        table = rng.choice("RST")
        x = rng.randrange(n_keys) + key_offset
        y = rng.randrange(n_keys) + key_offset
        target = sets[table]
        rec = (x, y)
        total = len(sets["R"]) + len(sets["S"]) + len(sets["T"])

        if rng.random() < 0.55:
            if rec in target:
                expected = False
            elif total >= B:
                # Input contract keeps |R|+|S|+|T| <= B; skip the op when the
                # model is already at capacity.
                continue
            else:
                target.add(rec)
                expected = True
            got = counter.insert(table, x, y)
        else:
            if rec in target:
                target.remove(rec)
                expected = True
            else:
                expected = False
            got = counter.delete(table, x, y)

        if got is not expected:
            raise AssertionError(
                "step %d: %s(%r,%r) changed=%r expected %r"
                % (step, table, x, y, got, expected)
            )

        if brute_every and step % brute_every == 0:
            want = count_triples(sets["R"], sets["S"], sets["T"])
            if counter.count() != want:
                raise AssertionError(
                    "step %d: count %d != enumeration %d (seed %d, B %d)"
                    % (step, counter.count(), want, seed, B)
                )
        if validate_every and step % validate_every == 0:
            counter.validate()
            want = count_triples(sets["R"], sets["S"], sets["T"])
            if counter.count() != want:
                raise AssertionError("step %d: count mismatch after validate" % step)
    return steps


def hub_scenario(seed: int, B: int = 20000) -> None:
    """Exercise promotion, heavy-pair memoisation and demotion.

    Three vertices, one per partition, get degree well above 2*sqrt(B); then
    their incident edges are churned so promotions/demotions really happen.
    """
    from triangle import Counter

    if B < 1000:
        raise ValueError("hub_scenario needs a larger B")
    rng = random.Random(seed)
    counter = Counter(B)
    sets: dict[str, Set[Pair]] = {"R": set(), "S": set(), "T": set()}

    def apply(table: str, x: int, y: int, insert: bool) -> bool:
        target = sets[table]
        rec = (x, y)
        if insert:
            if rec in target:
                return False
            if sum(map(len, sets.values())) >= B:
                return False
            target.add(rec)
            assert counter.insert(table, x, y) is True
            return True
        if rec not in target:
            assert counter.delete(table, x, y) is False
            return False
        target.remove(rec)
        assert counter.delete(table, x, y) is True
        return True

    # Hub key 0 in every partition; each hub sees ~k edges per side, so its
    # degree is ~2k and it becomes heavy (threshold 2*sqrt(B)).  Keep k modest
    # so the direct enumeration checks in this scenario stay cheap.
    k = min(200, max(50, B // 12))
    for i in range(1, k + 1):
        apply("R", 0, i, True)  # A-hub -- B_i
        apply("T", i, 0, True)  # C_i -- A-hub
        apply("R", i, 0, True)  # B-hub -- A_i
        apply("S", 0, i, True)  # B-hub -- C_i
        apply("S", i, 0, True)  # C-hub -- B_i
        apply("T", 0, i, True)  # C-hub -- A_i

    assert len(counter._heavy) >= 1, "hub should have been promoted"
    counter.validate()
    want = count_triples(sets["R"], sets["S"], sets["T"])
    if counter.count() != want:
        raise AssertionError("hub build: %d != %d" % (counter.count(), want))

    for step in range(4000):
        table = rng.choice("RST")
        if rng.random() < 0.5:
            x = rng.choice([0, 0, 0, rng.randrange(1, k + 1)])
            y = rng.randrange(1, 200)
        else:
            x = rng.randrange(k + 1)
            y = rng.randrange(1, 200)
        apply(table, x, y, rng.random() < 0.5)

        if step % 100 == 0:
            counter.validate()
            want = count_triples(sets["R"], sets["S"], sets["T"])
            if counter.count() != want:
                raise AssertionError(
                    "hub churn step %d: %d != %d"
                    % (step, counter.count(), want)
                )
    counter.validate()
