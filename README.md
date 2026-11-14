# Dynamic triangle counter — design notes

## 1. Model

Three relations are edges of a simple tripartite graph:

| record | edge |
|--------|------|
| `R(a,b)` | A–B |
| `S(b,c)` | B–C |
| `T(c,a)` | C–A |

A triple `(a,b,c)` satisfying all three is exactly an ordinary 3-cycle of
that graph, and conversely every 3-cycle is such a triple because the graph
has no intra-partition edges. So the requested count is the triangle count of
the tripartite graph.

Vertex identity: a key `k` used in `R` and the same integer used in `S` are
**different vertices**. `triangle.py` encodes a vertex as `3*k + side`
(`A=0`, `B=1`, `C=2`), which is injective for integer keys (Python ints are
arbitrary precision, so no overflow). Negative and huge keys work.

## 2. Data structures

`Counter` keeps:

* `_adj[u]` — adjacency set. Exactly one entry per live record (each undirected
  edge appears once in each endpoint's set), so total size `2|E|`.
* `_heavy` — set of heavy vertices.
* `_heavy_nbrs[u]` — `N(u) ∩ heavy`, maintained for **all** vertices
  (light ones included). Size `<= |E|`.
* `_hc[u][v]` (`u<v`) — memoised `|N(u) ∩ N(v)|` for pairs of heavy vertices
  only. Number of entries is `O(H^2)` where `H` is the number of heavy
  vertices.
* `_tri` — the answer; `_m` — number of records (also used to enforce `B`).

No list of triangles is ever stored.

## 3. Heavy/light thresholds

With `D = ceil(sqrt(B))`:

* a light vertex is promoted when its degree reaches `P = 2D`;
* a heavy vertex is demoted when its degree drops below `Q = D`.

The `P > Q` gap (hysteresis) is what makes the amortised bound work; a vertex
must lose `D` edges after promotion before it can be demoted again.

### Invariants (checked by `Counter.validate()`, used in every test)

1. Adjacency is symmetric, loop-free and contains no duplicates;
   `|E| == _m <= B`.
2. Every heavy vertex has degree `>= Q`; every light vertex has degree `< P`.
3. `_heavy_nbrs[u] == N(u) ∩ _heavy` for every vertex `u`.
4. `_hc` contains **exactly** the pairs of currently heavy vertices with
   nonzero common-neighbour counts, each equal to the true
   `|N(u) ∩ N(v)|` computed from adjacency.
5. `_tri` equals direct enumeration over the stored edges.

## 4. Update operation

Inserting edge `u–v` (deleting is symmetric):

1. **Count new triangles** = `|N(u) ∩ N(v)|`:
   * both heavy → `O(1)` lookup in `_hc`;
   * otherwise intersect the smaller adjacency set. If one endpoint is heavy,
     the other is light and has degree `< 2D`, so this is `O(D)`; if both are
     light both have degree `< 2D`, also `O(D)`.
2. **Link** `u–v` in `_adj`, and update `_heavy_nbrs` if an endpoint is heavy.
3. **Update `_hc`**: if one endpoint is heavy, every heavy neighbour `w` of the
   other endpoint gains one common neighbour → `O(#heavy neighbours)` which is
   `O(D)` because the other endpoint is light and at most half of `2D` of its
   neighbours can be heavy.
4. **Promote** an endpoint if its degree just reached `P`: add it to `_heavy`,
   register it in every neighbour's heavy set (`O(D)`), and materialise
   `_hc[u][x]` for all heavy `x` by intersecting adjacency sets — `O(D)` per
   heavy vertex, and there are at most `|E|/Q = O(B/D) = O(D)` heavy vertices,
   so `O(D^2) = O(B)`. Charged to the `P` new edges that caused promotion.
5. **Demote** an endpoint if its degree dropped below `Q`: discard its `_hc`
   row and its entries in other rows (`O(#heavy) = O(D)`), and remove it from
   neighbours' heavy sets (`O(D)`) — `O(D)` per demotion, and between two
   demotions of the same vertex at least `Q-P`… more precisely at least `Q`
   deletions happened, so the cost is amortised over the edges that paid for
   the promotion.

### Amortised cost

Let `D = ceil(sqrt(B))`. Write `H` for the number of heavy vertices. Since each
heavy vertex has degree `>= Q = D`, `H * D <= 2|E| <= 2B`, hence `H <= 2D`.

* Ordinary work per update: intersection `O(D)`, heavy-neighbour updates
  `O(H) = O(D)`. So **`O(D)` worst case per update**, no amortisation needed.
* Promotion: add to `_heavy`, register in neighbours' heavy sets `O(D)`, and
  intersect with all `H = O(D)` other heavy vertices at `O(D)` each → `O(D^2)`.
  A vertex can be promoted only once before it must shed `P - Q = D` edges to
  be demoted, so at most `O(B/D)` promotions occur over any `B`-edge history;
  total promotion work is `O((B/D) * D^2) = O(B * D)`, i.e. **`O(D)`
  amortised per operation**.
* Demotion: `O(D)` (discard its row, remove it from the other rows and from
  neighbours' heavy sets). At most one demotion per `D` deletions, so
  `O(D)` amortised as well.

Therefore `insert`/`delete` are **`O(sqrt(B))` amortised**, and the worst-case
single operation is `O(B)` only during a promotion. `count()` returns `_tri`,
so it is **`O(1)`** and touches no state (the `accesses` counter stays flat
across 200 000 calls in the benchmark).

### Space

`_adj` is `O(B)`; `_heavy_nbrs` is `O(B)`; `_hc` is at most
`O(H^2) <= O((B/Q)^2) = O(B)`. So `O(B)`.

## 5. Files

| file | purpose |
|------|---------|
| `triangle.py` | `Counter` (alias `TriangleCounter`), `validate()` |
| `reference.py` | independent enumerator + differential driver + hub scenario |
| `main.py` | CLI: `verify`, `demo`, `bench`, `help` |
| `bench.py` | hotspot / light workloads, access counts, time, RSS |
| `tests/test_triangle.py` | 17 unit, differential and heavy-light tests |

## 6. Independent enumeration

`reference.count_triples(R, S, T)` indexes `S` by its first component and, for
every `(a,b) ∈ R`, walks `S[a]` and tests `(c,a) ∈ T` — a completely separate
code path from the incremental counter. `verify` and the tests compare the two
after randomly generated insert/delete sequences (including duplicate inserts,
deletes of absent records, negative keys, and `B` as small as 4).

Sample results (all from the final code):

```
$ python3 main.py verify
example: R(1,2), S(2,3), T(3,1) -> 1; delete S(2,3) -> 0; restore -> 1
partition keys: equal integers in R/S/T form exactly one triple
differential: 12 runs x 400 ops vs independent enumeration, all equal
hub scenario: heavy vertices promoted, validated, counts equal
verify: all checks passed
```

The test suite additionally compares against the enumerator on dense
multi-hub graphs (two hubs per partition, `validate()` after **every**
operation) and on a scenario that promotes 40+ vertices at once and then
demotes half of them.

## 7. Measured costs

Command: `python3 main.py bench --B 200000 --ops 20000` (CPython 3.11,
single core, container). `accesses` counts every adjacency / heavy-table slot
read or written; "worst op" is the largest single-operation count. The two
workloads are run in separate processes so peak RSS is not cumulative.

```
== hotspot workload (hubs stay heavy; memoised intersections) ==
  [hotspot] B=200000: build 24997 records -> 38325 live (3 heavy), 26343 build accesses
    churn: 20000 ops (20000 changed record set), 20320 accesses, 1.60 us/op
    accesses/op: 1.0 (worst single op 4)   sqrt(B)=447   count()=6808   state entries=111626   peak RSS 33.4 MiB
== light workload (all degrees just under 2*sqrt(B)) ==
  [light] B=200000: build 99456 records -> 103987 live (0 heavy), 447 build accesses
    churn: 20000 ops (19889 changed record set), 10119313 accesses, 5.71 us/op
    accesses/op: 506.0 (worst single op 558)   sqrt(B)=447   count()=0   state entries=207974   peak RSS 63.8 MiB
== count() ==
  200000 calls in 0.009 s (0.044 us/call), 0 state accesses, checksum 200000
== scaling of the light workload (accesses per op) ==
           B    sqrt(B)  accesses/op      us/op    records        MiB
       10000        100        108.9       1.76       4975       14.2
       40000        200        213.5       2.63      21958       21.9
      160000        400        405.3       4.45       81983       52.9
      640000        800        798.6       7.86      321986      147.1
```

Interpretation:

* **Hot spot** (hub degree ≈ 8 300 ≫ `2 sqrt(B)` = 894): every churned hub
  edge is answered from the memoised heavy-pair table, so ~**1 access/op**,
  independent of `B`; worst single op 4 accesses.
* **Worst case** (all light, degree ≈ `2 sqrt(B)`): ~`sqrt(B)` accesses per
  changed record — 109 → 214 → 405 → 799 for `sqrt(B)` = 100 → 200 → 400 →
  800. This is exactly the predicted `O(sqrt(B))` behaviour (the constant is
  ≈ 1×`sqrt(B)` per operation because both endpoints are scanned).
* **`count()`** is a constant-time field read: 0 state accesses over 200 000
  calls, ~0.04 µs/call.
* **Memory** is linear in `B`: peak RSS 14 → 147 MiB while records grow
  5 000 → 322 000 (≈ 230 bytes/record measured on the light workload, which
  has two adjacency entries per record, plus interpreter baseline).
  `state_entries` (adjacency + heavy-neighbour + heavy-pair slots) is
  207 974 for 103 987 live records on the light workload — i.e. ≈ 2 per
  record, as predicted; the heavy table is empty there.
* Hot spot at `B = 50 000` (separate run): 8 328 churn ops, 10 408 accesses,
  1.2 accesses/op, worst op 4, 17.0 MiB — the memoised path is flat in `B`.

The "light" workload is deliberately adversarial (every update intersects two
sets of size ~`sqrt(B)`, and it is constructed to close zero triangles). Real
mixed workloads fall between the two tables.

