"""Stratified configuration-model null for cross-family target-list overlap.

Why this exists
---------------
The question is whether two families of abuse sources -- metadata-lure campaigns and address-poisoning
clusters -- target the same addresses more than they would if each picked targets independently. The
obvious approach, a risk ratio of P(poisoned | lure recipient) against a matched control, turned out to be
unidentifiable on this corpus: the outcome saturates the scored population (79-86% of scored candidate
transfers already reach the 4/4 threshold on the exploited chains), which confines any support-restricted
ratio to about [0, 1.16-1.26] and makes the comparison uninformative. Worse, every covariate cheap enough
to match on -- counts of transfers received or sent -- is either a descendant of the exposure or a
collider on it, so adjusting with them induces bias rather than removing it.

This module takes the other route. Instead of ADJUSTING for the confounder ("both channels prefer
addresses worth attacking"), it controls it by STRATIFICATION inside the null. The null holds fixed:

  - each source's target-list size within each stratum (so both list size and stratum mixture are fixed),
  - each stratum's address pool,

and then redraws each source's targets uniformly at random within each stratum.

Why address DEGREE is deliberately NOT preserved
------------------------------------------------
A degree-preserving configuration model is the obvious first instinct and it is provably useless here.
Write the statistic out: total cross-family overlap is

    O = sum over source pairs (a,b) of |L_a & L_b|  =  sum over addresses x of deg_A(x) * deg_B(x)

which is a DETERMINISTIC FUNCTION of the two per-family degree sequences. So any null that preserves
per-address degree preserves O exactly, giving O/E = 1.000 and zero power no matter how much sharing is
planted -- confirmed empirically before this docstring was written: a block of 40 addresses forced into
every source of both families still returned O/E = 1.0007. The reason is the same failure this analysis
has already hit twice at other levels: an address's abuse-degree is a DESCENDANT of the sharing under
test, so fixing it as a null constraint conditions the signal away.

The confounder must therefore be absorbed by strata, not by degree. Strata should be built from
attractiveness proxies (activity class, era, eligibility); within a stratum addresses are treated as
exchangeable. Residual within-stratum variation in attractiveness leaks into the statistic, which is why
the calibration arm is mandatory: a family of legitimate high-fan-out distributors must score O/E ~ 1
under the same strata, or the strata are too coarse and nothing may be published.

Sampler
-------
Within-stratum uniform resampling without replacement. For each source and stratum, the source's observed
count in that stratum is redrawn from that stratum's pool. This preserves per-source per-stratum list
sizes exactly and breaks any address-level pairing between the two families.

Inputs
------
Two families of target lists (sets of address strings) and a stratum label per address. No I/O, no
network, no global state.

Outputs
-------
`NullResult`: observed overlap, expected overlap under the null, the O/E ratio, a percentile interval and
a permutation p-value, plus the diagnostics needed to tell a real signal from a broken null.

Failure modes
-------------
- An empty family, or an address with no stratum label, RAISES. A silently dropped address would shrink a
  denominator and inflate the ratio.
- If any stratum contains fewer than two sources with distinct lists, no trade is possible there; the
  stratum is reported as frozen rather than silently contributing zero variance.
- If the observed statistic lies outside the null's support, the p-value is reported at its resolution
  bound (1/(n_perm+1)) and flagged, never as 0.

Complexity: O(n_perm x trades x mean list size). The trade count defaults to 5x the number of sources,
the mixing heuristic from the curveball literature.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NullResult:
    """Observed vs null cross-family overlap. Never quote `ratio` without `observed` and `expected`."""
    observed: int
    expected: float
    ratio: float | None
    ci95_low: float
    ci95_high: float
    p_value: float
    p_value_at_resolution_bound: bool
    n_permutations: int
    n_sources_a: int
    n_sources_b: int
    n_addresses: int
    n_universe: int
    n_strata: int
    frozen_strata: tuple[str, ...] = field(default_factory=tuple)
    saturated_strata: tuple[str, ...] = field(default_factory=tuple)
    free_source_stratum_cells: int = 0

    def summary(self) -> str:
        return (f"observed {self.observed} vs expected {self.expected:.1f} "
                f"(O/E {self.ratio if self.ratio is None else round(self.ratio, 3)}, "
                f"95% null interval [{self.ci95_low:.1f}, {self.ci95_high:.1f}], "
                f"p={self.p_value:.4g}{' at resolution bound' if self.p_value_at_resolution_bound else ''})")


def cross_overlap(family_a: dict[str, set[str]], family_b: dict[str, set[str]]) -> int:
    """Total shared addresses summed over every (a, b) source pair across the two families.

    Counts an address once per source pair that shares it, which is the quantity the configuration model
    randomises. A plain union-intersection would discard the multiplicity that carries the signal.
    """
    return sum(len(la & lb) for la in family_a.values() for lb in family_b.values())


def expected_overlap_closed_form(
    counts_a: list[dict[str, int]],
    counts_b: list[dict[str, int]],
    pool_sizes: dict[str, int],
) -> float:
    """E[cross-overlap] under the within-stratum-uniform null, in closed form.

    Why this exists: at corpus scale the permutation route is not affordable. Base carries 48 lure
    campaigns whose recipient lists run to tens of millions of addresses, so recomputing per-address
    degrees for every source in every one of 999 permutations is O(total list size x permutations) and
    hopeless. But the expectation is analytic, so it never needs simulating.

    Derivation. Cross-overlap is O = sum_x deg_A(x) * deg_B(x). Under the null each source redraws its
    k_{a,s} targets uniformly from stratum s's pool of size N_s, independently of every other source, so
    for an address x in stratum s:

        E[deg_A(x)] = sum over a of k_{a,s} / N_s = K_{A,s} / N_s

    and likewise for B. Because the two families are drawn independently, E[deg_A(x) deg_B(x)] factorises,
    and summing the N_s addresses of each stratum gives

        E[O] = sum over strata of  K_{A,s} * K_{B,s} / N_s

    which costs O(number of strata), not O(list size x permutations). `test_closed_form_matches_the_
    permutation_mean` checks it against the sampler on synthetic data, so the two routes are cross-
    validated rather than assumed equivalent.

    Note this is the expectation only. A percentile interval still needs permutations (or a moment
    approximation), but it can be estimated on a subsample or a coarsened universe while the POINT
    estimate stays exact.
    """
    if not pool_sizes:
        raise ValueError("no strata supplied")
    ka: dict[str, int] = defaultdict(int)
    kb: dict[str, int] = defaultdict(int)
    for c in counts_a:
        for s, k in c.items():
            ka[s] += k
    for c in counts_b:
        for s, k in c.items():
            kb[s] += k
    total = 0.0
    for s in set(ka) | set(kb):
        n = pool_sizes.get(s)
        if not n:
            raise ValueError(f"stratum {s!r} appears in a family but has no pool size; a missing pool "
                             f"would silently drop its contribution to the expectation")
        total += (ka[s] * kb[s]) / n
    return total


def per_stratum_counts(family: dict[str, set[str]], strata: dict[str, str]) -> list[dict[str, int]]:
    """Per-source, per-stratum target counts -- the sufficient statistic for the closed form."""
    out: list[dict[str, int]] = []
    for name in sorted(family):
        c: dict[str, int] = defaultdict(int)
        for a in family[name]:
            try:
                c[strata[a]] += 1
            except KeyError:
                raise ValueError(f"address {a!r} in source {name!r} has no stratum label") from None
        out.append(dict(c))
    return out


def _redraw_within_stratum(k: int, pool: list[int], rng: random.Random) -> set[int]:
    """Draw `k` distinct members of `pool` uniformly without replacement.

    Uniform within the stratum is the whole point: it is what makes addresses exchangeable given their
    attractiveness class, so the residual signal is coordination beyond that class. `rng.sample` is exact
    (no replacement) rather than k independent draws, which would collide and shrink the list.
    """
    if k > len(pool):
        raise ValueError(f"cannot draw {k} distinct addresses from a stratum pool of {len(pool)}; the "
                         f"stratum is smaller than a source's observed count in it, which means the "
                         f"strata and the family lists disagree")
    return set(rng.sample(pool, k))


def stratified_overlap_null(
    family_a: dict[str, set[str]],
    family_b: dict[str, set[str]],
    strata: dict[str, str],
    n_permutations: int = 999,
    seed: int = 42,
) -> NullResult:
    """Compare observed cross-family overlap against a stratified within-stratum-uniform null.

    `strata` is the ADDRESS UNIVERSE, not just the addresses that appear in the families: it must label
    every address that COULD have been targeted, i.e. the eligible background population, and it must be a
    superset of both families. This is not a convenience -- it is what gives the null anywhere to move.

    Deriving the stratum pools from the families alone (an earlier version did) makes the null permute
    only among already-targeted addresses. In the degenerate case where a shared block is the sole
    representative of its stratum, every source's count then equals the pool size, the sampler must
    reproduce the observed graph, and O/E collapses to 1 -- reported as "supports independence" when in
    fact no test occurred. `saturated_strata` and the no-freedom guard exist because that happened.

    A missing label raises rather than defaulting to a catch-all stratum, which would let addresses move
    between activity classes and relax the very constraint the null enforces.
    """
    if not family_a or not family_b:
        raise ValueError("both families must be non-empty; an empty family makes the overlap trivially "
                         "zero and would read as perfect independence")
    names_a = sorted(family_a)
    names_b = sorted(family_b)
    all_addr = set().union(*family_a.values(), *family_b.values())
    if not all_addr:
        raise ValueError("no addresses in either family")
    missing = {a for a in all_addr if a not in strata}
    if missing:
        raise ValueError(f"{len(missing)} addresses have no stratum label (e.g. "
                         f"{sorted(missing)[:3]}); refusing to default them into a catch-all stratum")

    # Pools come from the UNIVERSE (all of `strata`), not from the families, so the null can place a
    # source's targets on any eligible address of the same stratum.
    universe = sorted(strata)
    idx = {a: i for i, a in enumerate(universe)}
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for a in universe:
        by_stratum[strata[a]].append(idx[a])

    rows_all: list[set[int]] = ([{idx[a] for a in family_a[n]} for n in names_a] +
                                [{idx[a] for a in family_b[n]} for n in names_b])
    n_a = len(names_a)

    # Per-source, per-stratum counts: the invariant the null preserves.
    stratum_of_idx = {idx[a]: strata[a] for a in universe}
    counts: list[dict[str, int]] = []
    for row in rows_all:
        c: dict[str, int] = defaultdict(int)
        for i in row:
            c[stratum_of_idx[i]] += 1
        counts.append(dict(c))

    observed = cross_overlap(family_a, family_b)

    # Two degeneracies that both produce a spurious O/E ~ 1 and must never pass silently, because a
    # null with no freedom cannot distinguish "no sharing" from "no test".
    #
    #   frozen    -- at most one source is present in the stratum, so its contribution to the overlap is
    #                identical in every permutation.
    #   saturated -- some source's count equals the stratum's whole pool, so that source MUST redraw the
    #                identical set. Discovered the hard way: a planted 40-address shared block that
    #                coincided exactly with a 40-address stratum returned O/E = 1.03 with real sharing
    #                present, because the sampler had nothing left to choose.
    frozen = sorted(s for s, pool in by_stratum.items()
                    if sum(1 for c in counts if c.get(s)) < 2 or len(pool) < 2)
    saturated = sorted({s for c in counts for s, k in c.items() if k >= len(by_stratum[s])})
    free_pairs = sum(1 for c in counts for s, k in c.items()
                     if 0 < k < len(by_stratum[s]))
    if not free_pairs:
        raise ValueError(
            "the null has no freedom: every (source, stratum) cell is either empty or fills its whole "
            "stratum pool, so every permutation reproduces the observed graph and O/E is forced to 1. "
            "Use coarser strata, or larger stratum pools, before reading anything into the result.")

    rng = random.Random(seed)
    stats: list[int] = []
    for _ in range(n_permutations):
        work = [set() for _ in rows_all]
        for r, c in zip(work, counts):
            for s, k in c.items():
                r |= _redraw_within_stratum(k, by_stratum[s], rng)
        stats.append(sum(len(work[i] & work[n_a + j])
                         for i in range(n_a) for j in range(len(names_b))))

    stats.sort()
    expected = sum(stats) / len(stats)
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    # One-sided: how often does the null reach or exceed what we saw. +1 for the observed value itself.
    at_least = sum(1 for s in stats if s >= observed)
    p = (at_least + 1) / (n_permutations + 1)
    return NullResult(
        observed=observed,
        expected=expected,
        ratio=(observed / expected) if expected else None,
        ci95_low=float(lo), ci95_high=float(hi),
        p_value=p,
        p_value_at_resolution_bound=(at_least == 0),
        n_permutations=n_permutations,
        n_sources_a=len(names_a), n_sources_b=len(names_b),
        n_addresses=len(all_addr), n_universe=len(universe), n_strata=len(by_stratum),
        frozen_strata=tuple(sorted(frozen)),
        saturated_strata=tuple(saturated),
        free_source_stratum_cells=free_pairs,
    )
