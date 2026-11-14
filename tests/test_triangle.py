"""Unit and differential tests (standard library ``unittest`` only).

Run with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference import count_triples, run_differential  # noqa: E402
from triangle import Counter  # noqa: E402


class TestWorkedExample(unittest.TestCase):
    def test_statement_example(self):
        c = Counter(8)
        self.assertTrue(c.insert("R", 1, 2))
        self.assertTrue(c.insert("S", 2, 3))
        self.assertTrue(c.insert("T", 3, 1))
        self.assertEqual(c.count(), 1)
        self.assertTrue(c.delete("S", 2, 3))
        self.assertEqual(c.count(), 0)
        c.validate()

    def test_duplicate_insert_and_missing_delete(self):
        c = Counter(8)
        c.insert("R", 1, 2)
        self.assertFalse(c.insert("R", 1, 2))
        self.assertFalse(c.delete("S", 2, 3))
        self.assertEqual(c.count(), 0)
        self.assertTrue(c.delete("R", 1, 2))
        self.assertFalse(c.delete("R", 1, 2))
        c.validate()

    def test_empty_counter(self):
        c = Counter(1)
        self.assertEqual(c.count(), 0)
        self.assertFalse(c.delete("R", 0, 0))
        c.validate()

    def test_direction_matters(self):
        c = Counter(8)
        c.insert("R", 1, 2)
        c.insert("S", 2, 3)
        c.insert("T", 1, 3)  # T(c,a): needs T(3,1)
        self.assertEqual(c.count(), 0)
        c.insert("T", 3, 1)
        self.assertEqual(c.count(), 1)

    def test_partitions_are_distinct(self):
        # Equal integers in different tables are different vertices.
        c = Counter(8)
        c.insert("R", 0, 0)
        c.insert("S", 0, 0)
        c.insert("T", 0, 0)
        self.assertEqual(c.count(), 1)
        c.validate()

    def test_negative_and_large_keys(self):
        c = Counter(8)
        c.insert("R", -5, 10**12)
        c.insert("S", 10**12, -7)
        c.insert("T", -7, -5)
        self.assertEqual(c.count(), 1)
        c.validate()

    def test_invalid_table(self):
        c = Counter(4)
        with self.assertRaises(ValueError):
            c.insert("U", 1, 2)
        with self.assertRaises(ValueError):
            c.delete("X", 1, 2)

    def test_invalid_B(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                Counter(bad)
        for bad in (True, 1.5, "10"):
            with self.assertRaises(TypeError):
                Counter(bad)

    def test_capacity(self):
        c = Counter(2)
        c.insert("R", 1, 2)
        c.insert("S", 2, 3)
        with self.assertRaises(ValueError):
            c.insert("T", 3, 1)
        # A duplicate does not consume capacity and does not raise.
        self.assertFalse(c.insert("R", 1, 2))
        c.delete("R", 1, 2)
        self.assertTrue(c.insert("T", 3, 1))
        c.validate()

    def test_many_triangles(self):
        # Complete-ish structure: a in 0..2, b in 10..12, c in 20..22.
        c = Counter(100)
        for a in range(3):
            for b in range(10, 13):
                c.insert("R", a, b)
        for b in range(10, 13):
            for cc in range(20, 23):
                c.insert("S", b, cc)
        for cc in range(20, 23):
            for a in range(3):
                c.insert("T", cc, a)
        self.assertEqual(c.count(), 27)
        c.validate()
        c.delete("R", 1, 11)  # removes the three triples with b=11 and a=1
        self.assertEqual(c.count(), 27 - 3)
        c.validate()


class TestDifferential(unittest.TestCase):
    def test_random_small(self):
        for seed in range(8):
            run_differential(
                B=20 + seed,
                seed=seed,
                steps=500,
                n_keys=5 + seed % 4,
                validate_every=5,
                brute_every=1,
            )

    def test_tight_capacity(self):
        for seed in range(5):
            run_differential(
                B=4,
                seed=100 + seed,
                steps=300,
                n_keys=3,
                validate_every=3,
            )

    def test_enumerator_matches_naive(self):
        rng = random.Random(7)
        R = {(rng.randrange(4), rng.randrange(4)) for _ in range(8)}
        S = {(rng.randrange(4), rng.randrange(4)) for _ in range(8)}
        T = {(rng.randrange(4), rng.randrange(4)) for _ in range(8)}
        naive = sum(
            1
            for a in range(4)
            for b in range(4)
            for c in range(4)
            if (a, b) in R and (b, c) in S and (c, a) in T
        )
        self.assertEqual(count_triples(R, S, T), naive)

        c = Counter(64)
        for a, b in R:
            c.insert("R", a, b)
        for b, cc in S:
            c.insert("S", b, cc)
        for cc, a in T:
            c.insert("T", cc, a)
        self.assertEqual(c.count(), naive)
        c.validate()


class TestHeavyLight(unittest.TestCase):
    def test_promote_and_demote(self):
        c = Counter(50)  # D = 8, promote at 16, demote below 8
        for i in range(16):
            c.insert("R", 0, i)
        self.assertIn(0 * 3 + 0, c._heavy)  # A-key 0 is heavy
        for i in range(16):
            c.insert("S", i, 0)
        self.assertEqual(len(c._heavy), 2)
        c.validate()

        for i in range(7, 16):
            c.delete("R", 0, i)
        self.assertNotIn(0, c._heavy)
        c.validate()

        for _ in range(3):
            for i in range(7, 16):
                c.insert("R", 0, i)
            c.validate()
            for i in range(7, 16):
                c.delete("R", 0, i)
            c.validate()

    def test_heavy_pair_memo_is_exact(self):
        c = Counter(60)  # D = 8 (isqrt(60)=7 -> 8), P = 16
        for i in range(16):
            c.insert("R", 0, i)  # A0 heavy
            c.insert("R", 1, i)  # A1 heavy, shares all 16 neighbours
        for i in range(16):
            c.insert("S", i, 0)  # B_i -- C0
        c.validate()
        # Every R(0,i) + S(i,0) + T(0,0) plus R(1,i) + S(i,0) + T(0,?):
        # exactly the triples sharing the C0 vertex.
        c.insert("T", 0, 0)
        self.assertEqual(c.count(), 16)
        c.insert("T", 0, 1)  # C0 -- A1 also closes 16 triples
        self.assertEqual(c.count(), 32)
        # A0 and A1 share these B-neighbours plus C0 (from T(0,0)/T(0,1)).
        self.assertEqual(c._hc_get(0, 3), 17)
        c.validate()


class TestCountIsO1(unittest.TestCase):
    def test_count_does_not_touch_state(self):
        c = Counter(2000, track_accesses=True)
        for i in range(200):
            c.insert("R", 0, i)
            c.insert("S", i, 0)
            c.insert("T", 0, i)
        before = c.accesses
        for _ in range(5000):
            c.count()
        self.assertEqual(c.accesses, before)


class TestDenseHubs(unittest.TestCase):
    def test_two_hubs_per_partition(self):
        """Heavy-heavy edges exercise the memoised pair table directly."""
        rng = random.Random(3)
        B = 1500
        c = Counter(B)
        R, S, T = set(), set(), set()

        def apply(table, x, y, insert):
            target = {"R": R, "S": S, "T": T}[table]
            rec = (x, y)
            if insert:
                if rec in target:
                    self.assertFalse(c.insert(table, x, y))
                    return
                if len(R) + len(S) + len(T) >= B:
                    return
                target.add(rec)
                self.assertTrue(c.insert(table, x, y))
            else:
                if rec not in target:
                    self.assertFalse(c.delete(table, x, y))
                    return
                target.remove(rec)
                self.assertTrue(c.delete(table, x, y))

        for h in (0, 1):
            for i in range(90):
                for table, args in (
                    ("R", (h, i + 10)),
                    ("R", (i + 10, h)),
                    ("S", (i + 10, h)),
                    ("S", (h, i + 10)),
                    ("T", (i + 10, h)),
                    ("T", (h, i + 10)),
                ):
                    apply(table, *args, True)
        self.assertGreaterEqual(len(c._heavy), 4)

        for step in range(1500):
            table = rng.choice("RST")
            if rng.random() < 0.7:
                x = rng.choice([0, 1, rng.randrange(10, 100)])
                y = rng.choice([0, 1, 10 + rng.randrange(90)])
            else:
                x = rng.randrange(3)
                y = rng.randrange(100)
            apply(table, x, y, rng.random() < 0.5)
            c.validate()
            self.assertEqual(c.count(), count_triples(R, S, T))


class TestManyHeavy(unittest.TestCase):
    def test_mass_promotion_and_demotion(self):
        """Many heavy vertices at once; verify demotion rebuilds the table."""
        B = 20000
        c = Counter(B)
        R, S, T = set(), set(), set()

        def apply(table, x, y, insert):
            target = {"R": R, "S": S, "T": T}[table]
            rec = (x, y)
            if insert:
                if rec in target:
                    self.assertFalse(c.insert(table, x, y))
                    return
                if len(R) + len(S) + len(T) >= B:
                    return
                target.add(rec)
                self.assertTrue(c.insert(table, x, y))
            else:
                if rec not in target:
                    self.assertFalse(c.delete(table, x, y))
                    return
                target.remove(rec)
                self.assertTrue(c.delete(table, x, y))

        for a in range(1, 31):
            for b in range(1, 301):
                apply("R", a, b, True)
        for cc in range(1, 11):
            for a in range(1, 32):
                apply("T", cc, a, True)
            for b in range(1, 301):
                apply("S", b, cc, True)

        self.assertGreaterEqual(len(c._heavy), 40)
        c.validate()
        self.assertEqual(c.count(), count_triples(R, S, T))

        # Demote the T-heavy C vertices by dropping their S edges.
        for cc in range(6, 11):
            for b in range(1, 301):
                apply("S", b, cc, False)
        self.assertLess(len(c._heavy), 40)
        c.validate()
        self.assertEqual(c.count(), count_triples(R, S, T))


if __name__ == "__main__":
    unittest.main(verbosity=2)
