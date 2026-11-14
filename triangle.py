"""Dynamic counting of 3-coloured triangles.

The three relations are edges of a simple tripartite graph::

    R(a, b)  :  A -- B
    S(b, c)  :  B -- C
    T(c, a)  :  C -- A

A triple ``(a, b, c)`` is a closed cycle exactly when all three edges are
present.  Since the graph has no edges inside a partition, every ordinary
3-cycle is such a triple and vice versa.

The counter keeps the exact answer incrementally.  It uses the standard
degree/heavy-light dynamic triangle technique:

* light vertices are scanned when an incident edge changes;
* intersections for pairs of heavy vertices are memoised in a small table.

See README.md for the invariants and the amortised cost argument.
"""

from __future__ import annotations

import math
from typing import Dict, Set, Tuple

__all__ = ["Counter", "TriangleCounter"]

# Side numbers are encoded into vertex names together with the user supplied
# integer key.  The encoding is injective for integer keys and side in 0..2,
# so equal integers used in different relations stay distinct vertices.
_SIDES = {"R": (0, 1), "S": (1, 2), "T": (2, 0)}


def _vertex(value: int, side: int) -> int:
    return value * 3 + side


def _edge(table: str, x: int, y: int) -> Tuple[int, int]:
    try:
        left, right = _SIDES[table.upper()]
    except (AttributeError, KeyError):
        raise ValueError("table must be one of 'R', 'S' or 'T'") from None
    return _vertex(x, left), _vertex(y, right)


class Counter:
    """Maintain ``|{(a,b,c) : R(a,b), S(b,c), T(c,a)}|`` dynamically.

    Parameters
    ----------
    B:
        Positive upper bound on the total number of currently stored records
        (summed over R, S and T).  It is also used to choose the heavy/light
        threshold.

    Notes
    -----
    ``insert`` and ``delete`` need amortised ``O(sqrt(B))`` time; ``count`` is
    ``O(1)``.  The state is ``O(B)``.  No list of all triangles is stored.
    """

    __slots__ = (
        "B",
        "_D",
        "_P",
        "_Q",
        "_adj",
        "_heavy",
        "_heavy_nbrs",
        "_hc",
        "_tri",
        "_m",
        "_track",
        "accesses",
    )

    def __init__(self, B: int, *, track_accesses: bool = False) -> None:
        if isinstance(B, bool) or not isinstance(B, int):
            raise TypeError("B must be a positive integer")
        if B <= 0:
            raise ValueError("B must be a positive integer")

        self.B = B
        # D ~ sqrt(B).  A vertex is promoted at degree P=2D and demoted below
        # Q=D.  The gap gives hysteresis; see README for the amortisation.
        D = math.isqrt(B)
        if D * D != B:
            D += 1
        self._D = max(1, D)
        self._P = 2 * self._D
        self._Q = self._D

        # vertex -> set(neighbours)
        self._adj: Dict[int, Set[int]] = {}
        # current heavy vertices
        self._heavy: Set[int] = set()
        # vertex -> set of its neighbours that are heavy (maintained for all
        # vertices, including light ones)
        self._heavy_nbrs: Dict[int, Set[int]] = {}
        # heavy pair (u,v), u<v -> number of common neighbours
        self._hc: Dict[int, Dict[int, int]] = {}

        self._tri = 0
        self._m = 0
        self._track = track_accesses
        self.accesses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, table: str, x: int, y: int) -> bool:
        """Insert a record; return True iff the record set changed."""
        u, v = _edge(table, x, y)
        adj = self._adj
        su = adj.get(u)
        if su is not None and v in su:
            return False
        if self._m >= self.B:
            # The input contract guarantees |R|+|S|+|T| <= B, so this only
            # protects callers that would otherwise violate the bound.
            raise ValueError("record capacity B exceeded")

        new_triangles = self._common_count(u, v)
        self._add_edge(u, v)
        self._m += 1
        self._tri += new_triangles

        self._maybe_promote(u)
        self._maybe_promote(v)
        return True

    def delete(self, table: str, x: int, y: int) -> bool:
        """Delete a record; return True iff the record set changed."""
        u, v = _edge(table, x, y)
        adj = self._adj
        su = adj.get(u)
        if su is None or v not in su:
            return False

        removed_triangles = self._common_count(u, v)

        heavy = self._heavy
        hn = self._heavy_nbrs
        v_heavy = v in heavy
        u_heavy = u in heavy

        # The edge u-v is about to disappear.  First update the memoised
        # common-neighbour counts it currently contributes to.
        if v_heavy:
            nb = hn.get(u)
            if nb:
                for w in nb:
                    if w != v:
                        self._hc_add(w, v, -1)
        if u_heavy:
            nb = hn.get(v)
            if nb:
                for w in nb:
                    if w != u:
                        self._hc_add(w, u, -1)

        su.remove(v)
        adj[v].remove(u)
        if not su:
            del adj[u]
        if not adj[v]:
            del adj[v]

        if v_heavy:
            nb = hn.get(u)
            if nb is not None:
                nb.discard(v)
        if u_heavy:
            nb = hn.get(v)
            if nb is not None:
                nb.discard(u)

        self._m -= 1
        self._tri -= removed_triangles

        self._maybe_demote(u)
        self._maybe_demote(v)

        for z in (u, v):
            nb = hn.get(z)
            if nb is not None and not nb:
                del hn[z]
        return True

    def count(self) -> int:
        """Return the current number of closed R-S-T triples (O(1))."""
        return self._tri

    # ------------------------------------------------------------------
    # Common-neighbour queries
    # ------------------------------------------------------------------
    def _hc_get(self, u: int, v: int) -> int:
        if u > v:
            u, v = v, u
        row = self._hc.get(u)
        if self._track:
            self.accesses += 1
        return 0 if row is None else row.get(v, 0)

    def _hc_add(self, u: int, v: int, delta: int) -> None:
        if u > v:
            u, v = v, u
        row = self._hc.get(u)
        if row is None:
            if delta < 0:
                raise AssertionError("heavy-pair count underflow")
            row = {}
            self._hc[u] = row
            old = 0
        else:
            old = row.get(v, 0)
        new = old + delta
        if new < 0:
            raise AssertionError("heavy-pair count underflow")
        if new:
            row[v] = new
        else:
            row.pop(v, None)
            if not row:
                del self._hc[u]
        if self._track:
            self.accesses += 1

    def _hc_set(self, u: int, v: int, value: int) -> None:
        if u > v:
            u, v = v, u
        if value:
            self._hc.setdefault(u, {})[v] = value
        if self._track:
            self.accesses += 1

    def _common_count(self, u: int, v: int) -> int:
        """Exact |N(u) intersect N(v)| in the current graph."""
        if u in self._heavy and v in self._heavy:
            return self._hc_get(u, v)

        su = self._adj.get(u)
        sv = self._adj.get(v)
        if su is None or sv is None:
            return 0
        if len(su) > len(sv):
            su, sv = sv, su

        if not self._track:
            # Set intersection is implemented in C and iterates the smaller
            # operand.  By the invariants that operand is light (degree < 2D).
            return len(su & sv)

        cnt = 0
        for w in su:
            self.accesses += 1
            if w in sv:
                cnt += 1
        return cnt

    # ------------------------------------------------------------------
    # Edge maintenance
    # ------------------------------------------------------------------
    def _add_edge(self, u: int, v: int) -> None:
        """Insert u-v and update the heavy-pair table (not degrees/promote)."""
        heavy = self._heavy
        hn = self._heavy_nbrs
        v_heavy = v in heavy
        u_heavy = u in heavy

        # u becomes a new common neighbour of (v, w) for every heavy w
        # adjacent to u.  Symmetrically for v and neighbours of v.  (w == v
        # cannot occur here: insert rejects duplicates before this point, so v
        # is not yet in u's heavy-neighbour set.)
        if v_heavy:
            nb = hn.get(u)
            if nb:
                for w in nb:
                    self._hc_add(w, v, 1)
        if u_heavy:
            nb = hn.get(v)
            if nb:
                for w in nb:
                    self._hc_add(w, u, 1)

        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)

        if v_heavy:
            hn.setdefault(u, set()).add(v)
        if u_heavy:
            hn.setdefault(v, set()).add(u)

    def _maybe_promote(self, u: int) -> None:
        if u not in self._heavy and len(self._adj.get(u, ())) >= self._P:
            self._promote(u)

    def _promote(self, u: int) -> None:
        """Add u to the heavy set and materialise all heavy-pair rows."""
        adj_u = self._adj.get(u)
        if adj_u is None:
            return
        self._heavy.add(u)
        hn = self._heavy_nbrs
        for w in adj_u:
            hn.setdefault(w, set()).add(u)

        # u's own heavy-neighbour set was maintained while it was light and
        # therefore already contains exactly the heavy neighbours.
        for x in tuple(self._heavy):
            if x == u:
                continue
            adj_x = self._adj.get(x)
            if not adj_x:
                continue
            if self._track:
                cnt = 0
                for w in adj_u:
                    self.accesses += 1
                    if w in adj_x:
                        cnt += 1
            else:
                cnt = len(adj_u & adj_x)
            if cnt:
                self._hc_set(u, x, cnt)

    def _maybe_demote(self, u: int) -> None:
        if u in self._heavy and len(self._adj.get(u, ())) < self._Q:
            self._demote(u)

    def _demote(self, u: int) -> None:
        """Remove u from the heavy set and discard its memoised pairs."""
        self._heavy.discard(u)

        # Remove all heavy-pair entries involving u.  Canonical ordering puts
        # u either as the row key or as a key inside rows x<u.
        self._hc.pop(u, None)
        for x in self._heavy:
            if x < u:
                row = self._hc.get(x)
                if row is not None:
                    row.pop(u, None)
                    if not row:
                        del self._hc[x]

        # u is no longer a heavy neighbour of its neighbours.  Its own set of
        # heavy neighbours remains valid and is still needed by edge updates.
        hn = self._heavy_nbrs
        for w in self._adj.get(u, ()):
            nb = hn.get(w)
            if nb is not None:
                nb.discard(u)
                if not nb:
                    del hn[w]
        if self._track:
            self.accesses += len(self._heavy) + 1

    # ------------------------------------------------------------------
    # Optional invariant checking (used by the test suite)
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check all internal invariants.  Intended for tests/debugging."""
        B = self.B
        D, P, Q = self._D, self._P, self._Q
        if not (D >= 1 and P == 2 * D and Q == D):
            raise AssertionError("bad thresholds")

        # Adjacency is symmetric, loop-free, and within the edge budget.
        edges = set()
        for u, nbrs in self._adj.items():
            if not nbrs:
                raise AssertionError("empty adjacency entry")
            for v in nbrs:
                if v == u:
                    raise AssertionError("self loop")
                if u not in self._adj.get(v, ()):
                    raise AssertionError("asymmetric adjacency")
                edges.add((u, v) if u < v else (v, u))
        if 2 * len(edges) != sum(len(s) for s in self._adj.values()):
            raise AssertionError("bad degree sum")
        if len(edges) != self._m:
            raise AssertionError("edge count mismatch")
        if self._m > B:
            raise AssertionError("B exceeded")

        heavy = self._heavy
        # Hysteresis / threshold invariants.
        for u in heavy:
            if len(self._adj.get(u, ())) < Q:
                raise AssertionError("heavy vertex below demotion threshold")
        for u, nbrs in self._adj.items():
            if u not in heavy and len(nbrs) >= P:
                raise AssertionError("light vertex above promotion threshold")

        # Heavy-neighbour sets are exactly the heavy neighbours.
        for u, nbrs in self._adj.items():
            want = {v for v in nbrs if v in heavy}
            got = self._heavy_nbrs.get(u, set())
            if want != got:
                raise AssertionError("bad heavy-neighbour set")

        # Every pair of heavy vertices has an exact memoised count; no other
        # pairs may occur in the table.
        hlist = list(heavy)
        for i, u in enumerate(hlist):
            for v in hlist[i + 1 :]:
                want = len(self._adj.get(u, set()) & self._adj.get(v, set()))
                lo, hi = (u, v) if u < v else (v, u)
                got = 0
                row = self._hc.get(lo)
                if row is not None:
                    got = row.get(hi, 0)
                if got != want:
                    raise AssertionError(
                        "bad heavy pair %r,%r: %r != %r" % (u, v, got, want)
                    )
        for u, row in self._hc.items():
            if u not in heavy:
                raise AssertionError("stale heavy row")
            for v, val in row.items():
                if v not in heavy or u >= v or val <= 0:
                    raise AssertionError("bad heavy table entry")

        # The maintained triangle count equals direct enumeration.
        brute = 0
        # Partition of an encoded vertex is key % 3.
        for u, nbrs in self._adj.items():
            if u % 3 != 0:  # only start at A vertices
                continue
            for b in nbrs:
                for c in self._adj.get(b, ()):
                    if c % 3 == 2 and u in self._adj.get(c, ()):
                        brute += 1
        if brute != self._tri:
            raise AssertionError("triangle count %r != %r" % (self._tri, brute))


TriangleCounter = Counter
