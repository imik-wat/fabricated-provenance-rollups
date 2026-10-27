"""Matched controls for set-overlap claims: no overlap rate may be quoted without its base rate.

Why this exists
    The paper repeatedly wants to claim "recipients of X are also victims of Y". A raw overlap
    percentage cannot support that claim, because it is not compared against anything. This module
    exists because the omission was caught in this project once already: a *raw* 16.05% recipient
    overlap on Arbitrum looked publishable until a matched control drawn from the IDENTICAL parquet
    row groups showed the chance base rate was 10.82% — a ratio of only 1.48x. Repeating the
    measurement on Optimism gave 0.28% vs 0.60%, i.e. ratio 0.47x: the treatment group was LESS
    likely than chance, so the raw number would have been a fabrication with its sign inverted.
    The API therefore makes the control structural, not optional: ``overlap_ratio`` cannot be
    called without a control set, and ``OverlapResult.summary`` cannot render the ratio without
    printing both underlying rates next to it.

Inputs
    Address strings (three sets: treatment, control, reference), a bootstrap size and a seed.
    All sets are deduplicated and case-normalised; see ``_clean``.

Outputs
    A frozen ``OverlapResult`` carrying both raw rates, their ratio, a bootstrap 95% CI and an
    exact two-sided p-value. Nothing is written anywhere.

Failure modes
    Every unusable input RAISES rather than returning a sentinel a caller could mistake for a
    measurement: empty treatment / control / reference, non-string or blank elements, a treatment
    set that overlaps the control (then it is not a control), ``n_boot < 2``, a bootstrap whose
    replicates are all undefined, and in ``matched_control`` an eligible pool smaller than the
    requested size (a short control silently inflates the ratio, which is precisely the error class
    this module guards).

Complexity: O(|treatment| + |control| + |reference| + n_boot) time, O(|reference|) extra space.
Pure and deterministic: no network, no filesystem, no global state; all randomness comes from the
explicit ``seed``.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5


# --------------------------------------------------------------------------- input hygiene
def _clean(items: Iterable[str], name: str) -> set[str]:
    """Normalise an address collection to a deduplicated set.

    Purpose: make the three sets comparable. EVM addresses are case-insensitive (EIP-55 uses case
    only as a checksum), so a treatment set exported checksummed and a reference exported
    lowercase would intersect in ZERO elements and report a perfectly clean 0.0 overlap — a silent
    failure of exactly the class this repository's guards exist to catch. All three sets are
    therefore stripped and lowercased *identically*, so normalisation can never create or destroy
    an overlap asymmetrically.

    Inputs: any iterable of strings; ``name`` only for error messages.
    Outputs: a set of normalised strings. Deduplication is deliberate — an address that appears in
    a thousand transfers is one address, and counting it a thousand times would weight the rate by
    activity instead of by membership.
    Side effects: none.
    Failure modes: raises TypeError on a non-string element (a numpy/pandas NaN or bytes address
    would otherwise be silently stringified into a nonsense key), ValueError on a blank element or
    on an empty result.
    """
    out: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{name}: expected address strings, got {type(item).__name__!r}")
        norm = item.strip().lower()
        if not norm:
            raise ValueError(f"{name}: blank address string; an unresolvable id must not be dropped")
        out.add(norm)
    if not out:
        raise ValueError(f"{name}: empty set — an overlap statistic over nothing is not a "
                         "measurement, so this raises instead of returning 0")
    return out


# --------------------------------------------------------------------------- result container
@dataclass(frozen=True)
class OverlapResult:
    """One controlled overlap measurement. Immutable so a reported number cannot be edited later.

    ``ratio`` is meaningless on its own; ``summary()`` is the intended way to quote it because it
    always renders the two raw rates that produced it.
    """

    n_treatment: int
    hits_treatment: int
    rate_treatment: float
    n_control: int
    hits_control: int
    rate_control: float
    ratio: float
    ci95_low: float
    ci95_high: float
    p_value: float
    n_reference: int

    def summary(self) -> str:
        """Render the ratio WITH its base rates, so the ratio can never be quoted bare.

        Inputs: none. Outputs: a single-line string. Side effects: none. Failure modes: none.
        """
        return (
            f"ratio={self.ratio:.4g}x "
            f"[95% CI {self.ci95_low:.4g}, {self.ci95_high:.4g}] "
            f"= treatment {self.hits_treatment}/{self.n_treatment} "
            f"({self.rate_treatment:.4%}) / control {self.hits_control}/{self.n_control} "
            f"({self.rate_control:.4%}); p={self.p_value:.3g}; "
            f"|reference|={self.n_reference}"
        )

    def __str__(self) -> str:  # so print()/f-strings also carry the base rates
        return self.summary()


# --------------------------------------------------------------------------- the statistic
def overlap_ratio(treatment: Iterable[str],
                  control: Iterable[str],
                  reference: Iterable[str] | set[str],
                  n_boot: int = 1000,
                  seed: int = 42) -> OverlapResult:
    """Overlap rate of ``treatment`` with ``reference``, divided by the same rate for ``control``.

    Purpose: turn "16.05% of them are victims" into "1.48x chance", which is the only form of the
    claim the data supports.

    Inputs
        treatment, control : address collections, deduplicated and case-normalised, and required to
            be DISJOINT — a control that shares members with the treatment group is not a control,
            and the shared members would appear on both sides of the ratio.
        reference : the membership set both groups are tested against (e.g. known victims).
        n_boot : bootstrap replicates for the CI (>= 2).
        seed : the only source of randomness (project convention 42).

    Outputs
        ``OverlapResult``. ``rate_* = |group n reference| / |group|`` after deduplication;
        ``ratio = rate_treatment / rate_control``.

        ``ci95_*`` come from a nonparametric bootstrap that resamples the two groups
        INDEPENDENTLY, which is what their independent sampling justifies; a paired bootstrap would
        understate the variance. Because each group's statistic is the mean of 0/1 membership
        indicators, resampling |g| indicators with replacement makes the replicate hit count
        exactly Binomial(|g|, rate_g), so the replicates are drawn from that binomial rather than
        by materialising index arrays. This is an identity, not an approximation, and it keeps the
        cost O(n_boot) instead of O(n_boot * |g|).

        ``p_value`` is Fisher's exact test (two-sided) on the 2x2 table
        [[hits_t, n_t - hits_t], [hits_c, n_c - hits_c]]. Fisher rather than a chi-square or a
        normal approximation because (a) the null being tested is exactly Fisher's — reference
        membership is independent of which group an address is in, conditional on the margins —
        and (b) the interesting overlaps in this paper are sometimes tiny (28 hits in 10,000 on
        Optimism), where an asymptotic test is not trustworthy. Note the p-value tests
        ratio != 1, i.e. it says whether the excess is distinguishable from chance; it says
        nothing about whether the excess is large enough to matter. At rollup group sizes
        (>= ~200k addresses) it UNDERFLOWS to exactly 0.0; report that as "p < 1e-300", never as
        "p = 0".

    Side effects: none.

    Failure modes (all raise)
        - empty treatment / control / reference, or blank / non-string elements (see ``_clean``);
        - treatment and control not disjoint;
        - n_boot < 2 (no percentiles are definable);
        - rate_control == 0 AND rate_treatment == 0: the ratio is 0/0, carries no information at
          all, and there is no honest number to return, so this raises.
        - fewer than 2 usable bootstrap replicates.

        SPECIAL CASE, deliberately not an exception: rate_control == 0 while rate_treatment > 0.
        The ratio is then ``float('inf')``. This is chosen over raising because the measurement DID
        happen and is real ("the treatment group overlaps the reference; chance does not"), and
        over returning any finite number because an infinity cannot be mistaken for a plausible
        effect size — it propagates loudly through arithmetic and formats as "inf" in every table,
        so it cannot be quietly published. Treat it as "control too small to resolve the base
        rate" and enlarge the control.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot={n_boot}: need >= 2 replicates to define a percentile interval")

    t = _clean(treatment, "treatment")
    c = _clean(control, "control")
    ref = _clean(reference, "reference")

    shared = t & c
    if shared:
        raise ValueError(
            f"treatment and control share {len(shared)} address(es) (e.g. {sorted(shared)[0]}): "
            "a control must be disjoint from the treatment set, or the same addresses are counted "
            "on both sides of the ratio. Use matched_control(..., exclude=treatment).")

    n_t, n_c = len(t), len(c)
    hits_t, hits_c = len(t & ref), len(c & ref)
    rate_t, rate_c = hits_t / n_t, hits_c / n_c

    if rate_c == 0.0:
        if rate_t == 0.0:
            raise ValueError(
                "both treatment and control overlap the reference in 0 addresses: the ratio is "
                "0/0 and there is no measurement here. Widen the reference set or check that the "
                "three sets use the same address encoding.")
        ratio = float("inf")           # see the SPECIAL CASE note in the docstring
    else:
        ratio = rate_t / rate_c

    ci_low, ci_high = _bootstrap_ci(n_t, rate_t, n_c, rate_c, n_boot=n_boot, seed=seed)

    table = [[hits_t, n_t - hits_t], [hits_c, n_c - hits_c]]
    p_value = float(stats.fisher_exact(table, alternative="two-sided").pvalue)

    return OverlapResult(
        n_treatment=n_t, hits_treatment=hits_t, rate_treatment=rate_t,
        n_control=n_c, hits_control=hits_c, rate_control=rate_c,
        ratio=ratio, ci95_low=ci_low, ci95_high=ci_high,
        p_value=p_value, n_reference=len(ref),
    )


def _bootstrap_ci(n_t: int, rate_t: float, n_c: int, rate_c: float,
                  n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the ratio of two independent membership rates.

    Inputs: the two group sizes and observed rates, replicate count, seed.
    Outputs: (low, high) at the 2.5/97.5 percentiles; either bound may be ``inf``.
    Side effects: none.
    Failure modes: raises ValueError if fewer than 2 replicates are usable.

    Replicates in which the control draws zero hits are KEPT as ``inf`` rather than discarded:
    dropping them would quietly shift the upper bound down and make a fragile ratio look
    well-determined. Only the genuinely undefined 0/0 replicates are dropped, and their number is
    reported in the exception if too few remain.

    The percentiles use ``method="nearest"`` (an order statistic) rather than the default linear
    interpolation. Linear interpolation between two infinite neighbours evaluates inf - inf and
    returns a silent NaN — observed while writing this module, where a control with zero reference
    hits produced ratio=inf with a CI of (nan, nan), i.e. a missing measurement dressed as a
    present one. An order statistic is always a value the bootstrap actually produced; at
    n_boot >= 1000 it differs from the interpolated bound by at most one rank (0.1%).
    """
    rng = np.random.default_rng(seed)
    boot_t = rng.binomial(n_t, rate_t, size=n_boot) / n_t
    boot_c = rng.binomial(n_c, rate_c, size=n_boot) / n_c

    ratios = np.full(n_boot, np.nan)
    ok = boot_c > 0
    ratios[ok] = boot_t[ok] / boot_c[ok]
    # control drew 0 hits but treatment did not -> genuinely unbounded, keep as inf
    ratios[~ok & (boot_t > 0)] = np.inf
    usable = ratios[~np.isnan(ratios)]
    if usable.size < 2:
        raise ValueError(
            f"only {usable.size} of {n_boot} bootstrap replicates were defined (the rest drew 0 "
            "hits in BOTH groups): the sets are too small or the overlap too rare for an interval. "
            "Enlarge the groups rather than reporting a ratio without a CI.")

    low = float(np.percentile(usable, CI_LOW_PCT, method="nearest"))
    high = float(np.percentile(usable, CI_HIGH_PCT, method="nearest"))
    if math.isnan(low) or math.isnan(high):        # belt and braces: never emit a NaN interval
        raise ValueError(f"bootstrap produced a NaN interval ({low}, {high}) from {usable.size} "
                         "replicates; this is a bug, not a result")
    return low, high


# --------------------------------------------------------------------------- control construction
def matched_control(candidates: Sequence[str],
                    exclude: set[str],
                    size: int,
                    seed: int = 42) -> list[str]:
    """Draw a control set of exactly ``size`` addresses, uniformly, without replacement.

    Purpose: build the disjoint comparison group ``overlap_ratio`` demands. "Matched" here means
    the caller is responsible for drawing ``candidates`` from the same population as the treatment
    group (in this project: the identical parquet row groups, so that time window, token mix and
    activity level are held roughly constant). This function guarantees only the two properties it
    can see: uniformity and disjointness from ``exclude``.

    Inputs
        candidates : the eligible population (duplicates tolerated; deduplicated here).
        exclude : addresses to remove, normally the treatment set, so the two groups are disjoint.
        size : exact number of addresses to return; must be >= 1.
        seed : the only source of randomness.

    Outputs: a list of ``size`` distinct normalised addresses, none of them in ``exclude``.
        The eligible pool is SORTED before drawing so the draw depends only on the *set* of
        candidates and the seed — not on the row order the candidates happened to arrive in. Two
        runs over the same addresses read back in a different parquet order therefore produce the
        same control, which is what "reproducible" has to mean here.

    Side effects: none.

    Failure modes (all raise)
        - ``size < 1``; a zero-length control is not a control;
        - non-string / blank candidates, or an entirely empty candidate set (via ``_clean``);
        - eligible pool smaller than ``size``. This RAISES instead of returning a shorter list:
          a short control has a noisier and typically higher-variance base rate, so silently
          shrinking it inflates the apparent ratio — the exact failure this module exists to stop.
    """
    if size < 1:
        raise ValueError(f"size={size}: a control set must contain at least one address")

    pool_set = _clean(candidates, "candidates")
    excl = _clean(exclude, "exclude") if exclude else set()
    pool = sorted(pool_set - excl)

    if len(pool) < size:
        raise ValueError(
            f"eligible control pool has {len(pool)} addresses after removing {len(excl)} "
            f"excluded, but {size} were requested. Refusing to return a short control: it would "
            "inflate the ratio. Supply more candidates or reduce the treatment/control size.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=size, replace=False)
    return [pool[i] for i in idx]
