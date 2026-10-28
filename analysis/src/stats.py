"""Shared statistical estimators: promoted out of one-off scripts, plus the formal model's bounds.

Why this exists
    Five helpers lived as private functions inside individual scripts (`_wilson` in
    scripts/behavioural_detect.py, `_gini` in scripts/longitudinal.py, `_oof`/`_boot_f1`/`_mcnemar`
    in scripts/ablation.py). Being private they were untested and un-reusable, so every new script
    re-derived them -- exactly the copy-paste drift the project's silent-failure history warns
    about. They are promoted here with their semantics preserved and now tested. The scripts keep
    their own copies (this module adds no coupling and changes no published number); new code should
    import from here.

    The module also adds the three estimators the new paper's formal model needs
    (`rule_of_three`, `stratified_proportion`, `expected_jaccard`), each of which states a
    proposition, so the proposition's formula lives next to its implementation and its test.

Inputs
    Plain sequences / numpy arrays of counts, proportions, labels and predictions. Nothing is read
    from disk or the network; every table used is vendored in the source.

Outputs
    Pure Python scalars and tuples, or numpy arrays for the out-of-fold predictions. No file is
    written and no global state is touched, so results depend only on the arguments (plus `seed`
    where randomness is involved).

Failure modes
    Every function raises `ValueError` on an input for which its estimator is undefined -- n=0
    trials, negative wealth, weights that do not sum to one, a pool smaller than the sample drawn
    from it. This is deliberate and load-bearing: the project has a documented history of
    near-miss fabrications that survived because an undefined quantity was returned as 0.0 and read
    downstream as a real measurement. A zero returned here always means "measured zero".

Complexity: every function is O(len(input)) except `bootstrap_f1` (O(n_boot * len(y))) and
`oof_predictions` (k model fits, so dominated by the caller's estimator).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

# Wilson's default multiplier: the 97.5th percentile of the standard normal, i.e. a two-sided 95%
# interval. Kept as a named constant so a caller changing the level cannot mistake it for a fudge.
Z_95 = 1.96

# `stratified_proportion` tolerance on sum(W_h) == 1. Weights are population shares N_h/N, so they
# only fail to sum to one through a *missing stratum* or a typed number -- both of which must be
# loud. The tolerance therefore covers float summation error only, not sloppy bookkeeping.
WEIGHT_SUM_TOL = 1e-6


# --------------------------------------------------------------------------------- promoted
def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Purpose: interval for k successes in n trials that stays inside [0, 1] and remains sensible at
    k=0, k=n and small n, where the normal approximation p +/- z*sqrt(p(1-p)/n) does not (it gives a
    zero-width interval at k=0, which in this paper's tail estimates would understate a population
    extrapolation as exact).

        centre = (p + z^2/2n) / (1 + z^2/n),  half = z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)

    Inputs: k successes (0 <= k <= n), n trials (n > 0), z multiplier (> 0; 1.96 = 95%).
    Outputs: (lower, upper), clamped to [0, 1]. At k=0 the lower bound is exactly 0.0 and at k=n the
    upper bound is exactly 1.0 (the clamp absorbs the float residue of an algebraically exact
    bound).
    Side effects: none.
    Failure modes: raises ValueError for n <= 0, k outside [0, n], or z <= 0. The promoted original
    returned (0.0, 0.0) for n=0; that is indistinguishable from a measured-zero interval, so it is
    now an error (see the module's "Failure modes").
    """
    if n <= 0:
        raise ValueError(f"wilson: n must be > 0, got {n} (no trials => no interval, not (0, 0))")
    if not 0 <= k <= n:
        raise ValueError(f"wilson: need 0 <= k <= n, got k={k}, n={n}")
    if z <= 0:
        raise ValueError(f"wilson: z must be > 0, got {z}")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gini(values: Sequence[float] | np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, ->1 = one holder takes all).

    Purpose: concentration of loss across victims. Used to answer "is the reported total the sum of
    many small losses or one whale?", which changes what the number means.

        G = sum_i (2i - n - 1) * x_(i)  /  (n * sum_i x_i),   x_(i) sorted ascending, i = 1..n

    Inputs: a sequence of non-negative values (unsorted is fine; sorted internally).
    Outputs: G in [0, 1 - 1/n]. Two boundary cases are *defined*, not accidental:
      * all values equal (including all zero) -> 0.0. The formula's numerator is exactly 0 for an
        equal distribution, and the all-zero case (0/0) takes the same limit: no inequality is
        observable in a distribution with nothing to distribute.
      * the empty sequence -> ValueError, NOT 0.0. Zero victims is an absence of measurement, and
        0.0 would read in a table as "loss was perfectly evenly shared". The promoted original
        returned 0.0 here; that difference is intentional.
    Side effects: none.
    Failure modes: raises ValueError on an empty input, on any negative value (the Gini coefficient
    is only defined for non-negative quantities, and a negative entry can push G outside [0, 1]
    where it would silently look like a valid figure), or on NaN/inf.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"gini: expected a 1-D sequence, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("gini: undefined for an empty distribution (0.0 would read as 'equal')")
    if not np.all(np.isfinite(x)):
        raise ValueError("gini: input contains NaN or inf")
    if np.any(x < 0):
        raise ValueError(f"gini: negative values are not admissible (min={x.min()})")
    x = np.sort(x)
    n = x.size
    total = x.sum()
    if total == 0:
        return 0.0                      # all-zero == perfectly equal; see docstring
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def oof_predictions(
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scale: bool = False,
    k: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped out-of-fold predictions: every row predicted by a model that never saw its group.

    Purpose: honest generalisation estimates for the arbiter. Candidates are correlated within a
    victim (one victim contributes many lookalikes), so a random split leaks: the same victim's rows
    appear in train and test and the score measures memorisation. GroupKFold keeps each group
    (victim) wholly inside one fold.

    Inputs:
      make_model  zero-argument factory returning a fresh, unfitted estimator with .fit/.predict/
                  .predict_proba. A factory, not an instance: each fold must start untrained.
      X           (n_samples, n_features) float array.
      y           (n_samples,) labels.
      groups      (n_samples,) group key per row (e.g. victim address).
      scale       standardise features with statistics from the TRAINING fold only (needed by
                  distance/gradient models such as logistic regression). Fitting the scaler on all
                  rows would leak test-fold moments.
      k           number of folds (>= 2, <= number of distinct groups).
      seed        accepted for interface symmetry with the rest of this module and to document the
                  contract: fold construction here is *deterministic* (GroupKFold does not shuffle),
                  so the seed cannot change it. Estimator randomness must be seeded by the caller
                  inside `make_model` (as the scripts do, with random_state=42).
    Outputs: (pred, proba) -- hard predictions in y's dtype and P(class=1), each row filled from the
    single fold in which it was held out. Both are returned because the callers score PR-AUC from
    `proba`; returning only `pred` would lose that.
    Side effects: none here, but `make_model()` and .fit() are caller code and may use threads.
    Failure modes: raises ValueError on length mismatches, k < 2, an estimator without
    predict_proba, or -- the loud guard -- any row left unpredicted. The promoted original
    pre-filled `pred` with the sentinel -1 and never checked it; with unsigned labels that sentinel
    silently becomes 255, so coverage is now tracked in an explicit boolean mask and an incomplete
    cover raises. GroupKFold itself raises if k exceeds the number of groups.
    """
    # Imported lazily: scikit-learn is already pinned in requirements (the promoted original is a
    # thin wrapper over its GroupKFold/StandardScaler, and re-deriving the fold assignment here
    # would risk silently diverging from the splits the published results were computed on).
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError(f"oof_predictions: X must be 2-D, got shape {X.shape}")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"oof_predictions: length mismatch X={len(X)}, y={len(y)}, groups={len(groups)}")
    if len(y) == 0:
        raise ValueError("oof_predictions: no rows to predict")
    if k < 2:
        raise ValueError(f"oof_predictions: k must be >= 2, got {k}")

    pred = np.zeros(len(y), dtype=y.dtype)
    proba = np.zeros(len(y), dtype=float)
    filled = np.zeros(len(y), dtype=bool)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = make_model()
        if not hasattr(m, "predict_proba"):
            raise ValueError(
                f"oof_predictions: {type(m).__name__} has no predict_proba; a hard-label-only "
                "estimator cannot supply the probabilities the callers score PR-AUC from")
        m.fit(Xtr, y[tr])
        pred[te] = m.predict(Xte)
        proba[te] = m.predict_proba(Xte)[:, 1]
        filled[te] = True
    if not filled.all():
        raise ValueError(
            f"oof_predictions: {int((~filled).sum())} of {len(y)} rows were never held out; "
            "the fold cover is incomplete and the returned array is not out-of-fold")
    return pred, proba


def bootstrap_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the binary F1 score.

    Purpose: an F1 quoted to three decimals implies a precision the sample does not have. Resampling
    rows with replacement gives the interval within which the difference between two models is not
    resolvable, which is how the paper argues the ML arbiter does *not* beat the heuristic.

    Inputs: y_true / y_pred, equal-length sequences of 0/1 labels; n bootstrap replicates (>= 2);
    seed for the resampler.
    Outputs: (lo, hi), the 2.5th and 97.5th percentiles of the replicate F1s, unrounded (rounding is
    a presentation concern and belongs in the reporting script, not the estimator).
    Side effects: none. Randomness is a per-call `np.random.RandomState(seed)`, so the same
    arguments always give the same interval. The promoted original drew from a module-level
    RandomState shared with the SHAP subsample, which made its output depend on call order.
    Failure modes: raises ValueError on length mismatch, empty input, n < 2, or labels outside
    {0, 1} (F1 is computed for pos_label=1; {-1, +1} labels would score as if every -1 were a
    negative prediction and quietly return a wrong number).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError(
            f"bootstrap_f1: need equal-length 1-D arrays, got {yt.shape} and {yp.shape}")
    if yt.size == 0:
        raise ValueError("bootstrap_f1: no rows")
    if n < 2:
        raise ValueError(f"bootstrap_f1: need at least 2 replicates, got {n}")
    for name, arr in (("y_true", yt), ("y_pred", yp)):
        bad = np.setdiff1d(np.unique(arr), np.array([0, 1]))
        if bad.size:
            raise ValueError(f"bootstrap_f1: {name} must be 0/1, found {bad.tolist()}")

    rng = np.random.RandomState(seed)
    idx = np.arange(yt.size)
    f1s = np.empty(n, dtype=float)
    for i in range(n):
        s = rng.choice(idx, idx.size, replace=True)
        f1s[i] = f1_binary(yt[s], yp[s])
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return (float(lo), float(hi))


def mcnemar(
    y_true: Sequence[Any] | np.ndarray,
    pred_a: Sequence[Any] | np.ndarray,
    pred_b: Sequence[Any] | np.ndarray,
) -> tuple[float, float]:
    """Exact McNemar test: do two predictors differ on the rows where they disagree?

    Purpose: comparing two models by their headline F1 on the same rows is not a test -- the samples
    are paired. McNemar conditions on the discordant rows only: n01 = A wrong / B right,
    n10 = A right / B wrong, and asks whether a discordance is equally likely to fall either way.

        p = Binomial(min(n01, n10); n01 + n10, 1/2), two-sided exact

    Inputs: y_true and the two predictors' predictions, all equal length. Labels may be of any type
    that compares with ==; only agreement with y_true is used.
    Outputs: (statistic, p_value) where statistic = min(n01, n10), the exact test's statistic. The
    chi-square form (|n01-n10|-1)^2/(n01+n10) is deliberately NOT returned: the promoted original
    uses the exact binomial test, which is the right choice when the discordant count is small, and
    mixing the two statistics across scripts would make results incomparable.
    Side effects: none.
    Failure modes: raises ValueError on length mismatch or empty input. With no discordant rows at
    all (identical predictors) the p-value is 1.0 by definition -- there is no evidence of a
    difference -- and this is a genuine result, not a fallback.
    """
    from scipy.stats import binomtest             # scipy import kept local, as in the original

    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    if not (yt.shape == pa.shape == pb.shape) or yt.ndim != 1:
        raise ValueError(
            f"mcnemar: need three equal-length 1-D arrays, got {yt.shape}, {pa.shape}, {pb.shape}")
    if yt.size == 0:
        raise ValueError("mcnemar: no rows")
    n01 = int(np.sum((pa != yt) & (pb == yt)))
    n10 = int(np.sum((pa == yt) & (pb != yt)))
    stat = float(min(n01, n10))
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return (stat, float(p))


# ------------------------------------------------------------------- new: formal-model estimators
def rule_of_three(n: int, conf: float = 0.95) -> float:
    """Upper confidence bound on an event probability after ZERO events in n trials (Proposition 1).

    Purpose: the ZK chains show 0 successful poisoning attacks. "0" is not "impossible": with n
    observed opportunities and no success, the largest rate consistent with the data at confidence
    `conf` is the p for which the probability of seeing no event is exactly 1 - conf:

        (1 - p)^n = 1 - conf   =>   p_max = 1 - (1 - conf)^(1/n)

    The familiar "rule of three", p_max ~= 3/n, is the conf=0.95 first-order approximation
    (-ln 0.05 = 2.996). This function returns the EXACT form, which is what the tests pin: at n=50
    the exact bound is 0.0582 while 3/n is 0.0600, so returning the literal 3/n would disagree with
    the exact form at three decimals. The approximation is always the more conservative (larger) of
    the two, so quoting it in prose remains safe.

    Inputs: n > 0 trials with zero events; 0 < conf < 1.
    Outputs: the upper bound as a proportion in (0, 1).
    Side effects: none.
    Failure modes: raises ValueError for n <= 0 (no trials bound nothing: an unobserved chain is not
    a safe chain, and 0.0 here would assert exactly that) or conf outside (0, 1).
    """
    if n <= 0:
        raise ValueError(f"rule_of_three: n must be > 0, got {n} (zero trials bound nothing)")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"rule_of_three: conf must be in (0, 1), got {conf}")
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def stratified_exact_upper_bound(
    strata: Sequence[tuple[int, int, int]], conf: float = 0.95,
) -> tuple[int, float, tuple[int, ...]]:
    """Exact upper confidence bound on a population count under stratified sampling WITHOUT
    replacement (Proposition 2, finite-population clause).

    Purpose: `rule_of_three` bounds a rate after zero events, but it is a binomial bound -- it
    assumes independent draws with replacement from an infinite population, ignores the design
    weights, and ignores the finite-population correction. Applying it to the pooled count of a
    STRATIFIED without-replacement sample is not a design-based bound, which is exactly the
    objection this function answers. Here each stratum h has a KNOWN population `M_h`, a sample of
    `m_h` drawn without replacement, and `x_h` observed events; the bound is derived by inverting
    the exact hypergeometric test.

    Construction. Conditionally on the realised per-stratum sample sizes, the strata are drawn
    independently and X_h ~ Hypergeom(M_h, K_h, m_h). Conditioning is legitimate because the law of
    m_h involves M_h and the stratum sizes but NOT K, so m is ancillary for K. The test statistic is
    the TOTAL T = sum_h X_h, and for a hypothesised vector K the p-value is the lower tail

        p(K) = P_K( T <= sum_h x_h ),      T a convolution of independent hypergeometrics.

    The confidence region is {K : p(K) > 1 - conf}. Each member of that family of tests has size at
    most 1 - conf, because a discrete lower-tail p-value satisfies P(p <= a) <= a; inverting the
    family gives a region covering the true K with probability at least `conf`. The region is then
    intersected with the box x_h <= K_h <= M_h - m_h + x_h. The data place the true K inside that
    box with certainty, so the intersection keeps the region's coverage, and it removes allocations
    the observation has already ruled out. A confidence region for K bounds EVERY functional of K
    evaluated over it, so the maximum of sum_h K_h over the region is a valid one-sided upper
    bound. The result is exact, not asymptotic, and assumes nothing beyond the sampling design.

    Why the total and not the product of per-stratum tails: `prod_h P(X_h <= x_h)` is NOT a p-value
    when any x_h > 0 -- a product of independent p-values is not itself a p-value, and using one
    would make the bound anti-conservative. The two coincide only at x = 0, where the orthant
    {X = 0} IS the event {T <= 0}. The general form is used throughout so the zero case is not a
    special path that could silently diverge.

    Two properties make the result checkable, and both are asserted by tests:
      * a stratum sampled in full (`m_h == M_h`) contributes exactly zero, because an event there
        could not have been missed -- this is the finite-population correction at its limit;
      * with one stratum, no observed events and `M >> m`, the bound approaches the `rule_of_three`
        value, so the binomial rule is recovered as the infinite-population special case.

    Inputs: `strata` as (M_h, m_h, x_h) triples -- stratum population, sample size, observed events;
    `conf` in (0, 1).
    Outputs: (max_total_events, max_total_events / sum(M_h), per-stratum allocation achieving it).
    Side effects: none.
    Complexity: the search is over prod_h (M_h - m_h + 1) vectors, pruned by a running product, so
    it is cheap while the unsampled remainders are small. It is exact, not a heuristic: the
    per-stratum probability is non-increasing in K_h, so the branch bound below is sound.
    Failure modes: raises ValueError on an empty design, `M_h <= 0`, `m_h` outside [1, M_h], `x_h`
    outside [0, m_h], or `conf` outside (0, 1). A design in which every stratum is fully enumerated
    legitimately returns 0 -- that is a census, not a failure.
    """
    from scipy.stats import hypergeom          # scipy import kept local, as in `mcnemar`

    if not strata:
        raise ValueError("stratified_exact_upper_bound: no strata given; an empty design bounds "
                         "nothing and returning 0 would assert certainty")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"stratified_exact_upper_bound: conf must be in (0, 1), got {conf}")
    for i, (big_m, m, x) in enumerate(strata):
        if big_m <= 0:
            raise ValueError(f"stratified_exact_upper_bound: stratum {i} has M_h={big_m}; a stratum "
                             f"must contain at least one unit")
        # m_h == 0 is legitimate and consequential: a stratum of the subpopulation that was never
        # sampled constrains nothing, so it contributes its whole M_h to the bound. Rejecting it
        # would silently drop the least-observed part of the frame, which is the part most able to
        # hide events.
        if not 0 <= m <= big_m:
            raise ValueError(f"stratified_exact_upper_bound: stratum {i} has m_h={m} outside "
                             f"[0, {big_m}]; a sample cannot exceed its stratum")
        if not 0 <= x <= m:
            raise ValueError(f"stratified_exact_upper_bound: stratum {i} observed x_h={x} outside "
                             f"[0, {m}]; more events than units drawn is impossible")

    alpha = 1.0 - conf
    t_obs = sum(x for _, _, x in strata)

    # Per stratum and per admissible K_h, the pmf of X_h truncated at t_obs (mass above t_obs can
    # never contribute to P(T <= t_obs)). The data fix K_h on both sides with certainty:
    #   * K_h <= M_h - m_h + x_h, since the events not drawn are at most the units not drawn;
    #   * K_h >= x_h, since x_h events were seen in the stratum.
    # The region is the test-inversion region intersected with this box. Because the box contains
    # the true K with probability one, the intersection keeps the coverage of the region. Omitting
    # the lower side (an earlier version enumerated K_h from 0) was still valid but not tight. It
    # admitted, for instance, K_h = 0 in a fully enumerated stratum where an event had been seen.
    # The tail mass that freed then let other strata take more events, so the worst-case detector
    # bound came out 15/141 where the data-consistent maximum is 13/141.
    # Each entry is (K_h, pmf), so the K value is explicit and never inferred from a list index.
    pmfs: list[list[tuple[int, np.ndarray]]] = []
    for big_m, m, x in strata:
        k_max = big_m - m + x
        pmfs.append([(k, np.asarray(hypergeom.pmf(np.arange(t_obs + 1), big_m, k, m), dtype=float))
                     for k in range(x, k_max + 1)])

    def _cdf(pmf: np.ndarray) -> float:
        return float(pmf.sum())

    # Dynamic programme over strata. A state is (cumulative K, truncated pmf of the partial total).
    # Two states with the same cumulative K are comparable: if one has a pointwise-larger cumulative
    # distribution it can never lose, because the final tail probability is a NON-NEGATIVE weighted
    # combination of those cumulative entries. Dominated states are therefore discarded exactly,
    # not heuristically.
    front: dict[int, list[np.ndarray]] = {0: [np.eye(1, t_obs + 1, 0, dtype=float)[0]]}

    for table in pmfs:
        nxt: dict[int, list[np.ndarray]] = {}
        for k_cum, states in front.items():
            for pmf in states:
                for k, add in table:
                    conv = np.convolve(pmf, add)[:t_obs + 1]
                    if float(conv.sum()) <= alpha:
                        continue     # the tail only shrinks under further convolution: prune now
                    bucket = nxt.setdefault(k_cum + k, [])
                    cum = np.cumsum(conv)
                    keep = True
                    for j in range(len(bucket) - 1, -1, -1):
                        other = np.cumsum(bucket[j])
                        if np.all(other >= cum - 1e-18):
                            keep = False
                            break
                        if np.all(cum >= other - 1e-18):
                            bucket.pop(j)
                    if keep:
                        bucket.append(conv)
        front = nxt

    feasible = [k for k, states in front.items() if any(_cdf(s) > alpha for s in states)]
    if not feasible:                    # unreachable: K = x gives P(T <= t_obs) = 1
        raise ValueError("stratified_exact_upper_bound: no admissible allocation; the observed "
                         "outcome is impossible under every hypothesis, which cannot happen")
    best_total = max(feasible)

    # Recover one achieving allocation by re-running the search restricted to that total.
    alloc = _recover_allocation(pmfs, best_total, t_obs, alpha)
    return best_total, best_total / sum(big_m for big_m, _, _ in strata), alloc


def _recover_allocation(pmfs, target: int, t_obs: int, alpha: float) -> tuple[int, ...]:
    """Find one allocation summing to `target` whose tail probability exceeds `alpha`.

    Separated from the bound itself so the reported allocation is verified against the same
    criterion rather than carried along by fragile back-pointers through the dominance pruning.
    """
    n = len(pmfs)
    suffix_min = [0] * (n + 1)
    suffix_max = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + pmfs[i][0][0]
        suffix_max[i] = suffix_max[i + 1] + pmfs[i][-1][0]
    out: list[int] | None = None

    def walk(i: int, remaining: int, pmf: np.ndarray, acc: list[int]) -> bool:
        nonlocal out
        if remaining < suffix_min[i] or remaining > suffix_max[i]:
            return False
        if i == n:
            if remaining == 0 and float(pmf.sum()) > alpha:
                out = list(acc)
                return True
            return False
        for k, add in reversed(pmfs[i]):
            if k > remaining:
                continue
            conv = np.convolve(pmf, add)[:t_obs + 1]
            if float(conv.sum()) <= alpha:
                continue                       # tail only shrinks under further convolution
            if walk(i + 1, remaining - k, conv, acc + [k]):
                return True
        return False

    walk(0, target, np.eye(1, t_obs + 1, 0, dtype=float)[0], [])
    if out is None:
        raise ValueError("stratified_exact_upper_bound: the bound was attained but no allocation "
                         "reproduces it; the dominance pruning and the search disagree")
    return tuple(out)


def stratified_proportion(strata: Sequence[tuple[float, int, int, int]]) -> tuple[float, float]:
    """Proportion and standard error from a stratified sample, with finite-population correction
    (Proposition 2).

    Purpose: the labelled sets are not simple random samples -- low match levels are subsampled
    heavily, high levels are a census. Pooling the hits would weight the estimate by *sampling
    effort* instead of population share. The stratified estimator reweights each stratum by its
    population share W_h = N_h / N:

        p_hat = sum_h W_h * p_h,          p_h = k_h / n_h
        Var   = sum_h W_h^2 * p_h(1 - p_h)/n_h * (1 - n_h/N_h)

    The trailing (1 - n_h/N_h) is the finite-population correction. It matters here rather than
    being a textbook nicety: strata that are a complete census (n_h = N_h) contribute exactly zero
    variance, because nothing about them was left to chance. Omitting the FPC overstates the
    variance -- it prices in sampling noise for rows that were all observed -- and the overstatement
    grows with the sampled fraction, i.e. precisely on this paper's censused high-level strata.

    Inputs: one 4-tuple per stratum, (W_h, k_h, n_h, N_h): population weight, hits observed, rows
    sampled, stratum population. The 4-tuple (rather than the 3-tuple (W, k, n)) is required because
    N_h cannot be recovered from W_h alone without also knowing N, and the FPC needs it explicitly.
    Outputs: (p_hat, standard_error), both plain floats.
    Side effects: none.
    Failure modes: raises ValueError if the stratum list is empty, if sum(W_h) differs from 1 by
    more than WEIGHT_SUM_TOL (=1e-6; a short sum means a stratum was forgotten, which would bias
    p_hat downwards while still looking like a valid proportion), if any W_h < 0, if n_h <= 0
    (an unsampled stratum has no p_h -- it must be dropped from the frame and the weights
    renormalised deliberately, not defaulted to zero here), if n_h > N_h, or if k_h is outside
    [0, n_h].
    """
    if len(strata) == 0:
        raise ValueError("stratified_proportion: no strata")
    w_sum = 0.0
    p_hat = 0.0
    var = 0.0
    for h, row in enumerate(strata):
        if len(row) != 4:
            raise ValueError(
                f"stratified_proportion: stratum {h} has {len(row)} fields, expected 4 "
                "(W_h, k_h, n_h, N_h)")
        w, k, n, big_n = float(row[0]), int(row[1]), int(row[2]), int(row[3])
        if w < 0:
            raise ValueError(f"stratified_proportion: stratum {h} has negative weight {w}")
        if n <= 0:
            raise ValueError(f"stratified_proportion: stratum {h} has n_h={n}; nothing was sampled")
        if big_n < n:
            raise ValueError(
                f"stratified_proportion: stratum {h} sampled n_h={n} from N_h={big_n} rows")
        if not 0 <= k <= n:
            raise ValueError(f"stratified_proportion: stratum {h} has k_h={k} outside [0, {n}]")
        p = k / n
        w_sum += w
        p_hat += w * p
        var += w * w * p * (1.0 - p) / n * (1.0 - n / big_n)
    if abs(w_sum - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(
            f"stratified_proportion: weights sum to {w_sum!r}, not 1 "
            f"(tolerance {WEIGHT_SUM_TOL}); a missing stratum biases p_hat")
    return (float(p_hat), float(math.sqrt(var)))


def expected_jaccard(k1: int, k2: int, n_pool: int) -> float:
    """Expected Jaccard of two INDEPENDENT uniform samples from a shared pool (Proposition 3).

    Purpose: the null for an observed overlap. Two attacker sets drawn from the same pool of
    addresses overlap a little by chance, so "the sets are disjoint" is only a finding relative to
    how much chance overlap was expected. For sizes k1, k2 from a pool of n_pool,

        E|A n B| = k1 * k2 / n_pool
        E[J]     = E[ |A n B| / |A u B| ]  ~=  E|A n B| / E|A u B|
                 = (k1*k2/n_pool) / (k1 + k2 - k1*k2/n_pool)  ~=  k1*k2 / (n_pool*(k1 + k2))

    Two approximations are deliberate and stated so they are not mistaken for exactness: the
    expectation of the ratio is approximated by the ratio of the expectations (they differ at
    O(1/k), negligible when the expected intersection is far below k1 + k2), and the union is taken
    as k1 + k2, dropping the expected intersection from the denominator. The second *understates*
    E[J] by the factor (1 - E|A n B|/(k1 + k2)) -- so the returned value is a slight under-estimate
    of the true null, which is the safe direction for the paper's argument (the real chance overlap
    is if anything larger than quoted). Both approximations require k1*k2/n_pool << k1 + k2; outside
    that regime the formula is not usable (at k1 = k2 = n_pool the two samples are the same set,
    true J = 1, and this returns 0.5 -- pinned in the tests as a documented limit).

    This is the value against which an observed Jaccard of 0.000 must be judged: when the null is
    itself ~1e-4, a measured zero is consistent with independence and is NOT evidence of separate
    attacker populations.

    Inputs: k1, k2 >= 0 sample sizes, n_pool > 0 pool size with n_pool >= max(k1, k2).
    Outputs: the expected Jaccard index, a float in [0, 1].
    Side effects: none.
    Failure modes: raises ValueError if n_pool < max(k1, k2) (a sample cannot exceed the pool it was
    drawn from -- silently accepting it would return an "expected overlap" above 1 or, worse, a
    plausible-looking small number computed from an inconsistent frame), if n_pool <= 0, if either
    k is negative, or if k1 + k2 == 0 (two empty samples have an undefined Jaccard, 0/0, and 0.0
    would read as "measured no overlap").
    """
    if k1 < 0 or k2 < 0:
        raise ValueError(f"expected_jaccard: sample sizes must be >= 0, got {k1}, {k2}")
    if n_pool <= 0:
        raise ValueError(f"expected_jaccard: n_pool must be > 0, got {n_pool}")
    if n_pool < max(k1, k2):
        raise ValueError(
            f"expected_jaccard: n_pool={n_pool} is smaller than the largest sample "
            f"({max(k1, k2)}); the samples and the pool come from inconsistent frames")
    if k1 + k2 == 0:
        raise ValueError("expected_jaccard: both samples are empty; Jaccard is undefined (0/0)")
    return float(k1 * k2) / (float(n_pool) * float(k1 + k2))


# ------------------------------------------------------------------------------------- helper
def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 for pos_label=1, matching sklearn's f1_score(..., zero_division=0).

    Purpose: `bootstrap_f1` calls this once per replicate, so it is kept dependency-free and
    allocation-free rather than importing scikit-learn for three counts.
    Inputs: equal-length 0/1 arrays (validated by the caller).
    Outputs: F1 in [0, 1]; 0.0 when the score is 0/0, i.e. a replicate in which the predictor made
    no positive prediction AND there was no positive to find. That zero is sklearn's
    zero_division=0 convention and is a *defined* score for a degenerate resample, not a
    swallowed error; it is why the bootstrap of a rare-positive class shows a lower tail at 0.
    Side effects: none. Failure modes: none for valid 0/1 input.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def cohen_kappa(
    a: Sequence[object], b: Sequence[object], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float, float]:
    """Cohen's kappa for two raters, with a bootstrap 95% interval.

    Returns (kappa, ci_low, ci_high, observed_agreement).

    kappa = (po - pe) / (1 - pe), where po is the observed agreement rate and pe the agreement expected
    from the raters' marginal distributions alone. It answers "how much better than chance do these two
    raters agree", which raw agreement cannot: two raters who both call 95% of items `legitimate` agree
    90% of the time by accident.

    Interval by bootstrap over ITEMS rather than the asymptotic formula, because the asymptotic variance is
    unreliable for small samples and for skewed marginals -- and skewed marginals are exactly the situation
    here (one class dominates a lure-annotation sample).

    Failure modes, all loud:
      - Unequal lengths, or fewer than 2 items, raise.
      - Perfect agreement with a SINGLE class used by both raters leaves pe == 1, so kappa is undefined
        (0/0) rather than 1.0; that raises, because reporting 1.0 there would claim reliability from a
        sample that demonstrates none.

    Complexity: O(n_boot * n).
    """
    import random as _random

    if len(a) != len(b):
        raise ValueError(f"raters disagree in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        raise ValueError("need at least 2 items for kappa")

    def _k(idx: Sequence[int]) -> float | None:
        m = len(idx)
        agree = sum(1 for i in idx if a[i] == b[i])
        po = agree / m
        ca: dict[object, int] = {}
        cb: dict[object, int] = {}
        for i in idx:
            ca[a[i]] = ca.get(a[i], 0) + 1
            cb[b[i]] = cb.get(b[i], 0) + 1
        pe = sum((ca.get(k, 0) / m) * (cb.get(k, 0) / m) for k in set(ca) | set(cb))
        if pe >= 1.0:
            return None
        return (po - pe) / (1.0 - pe)

    full = list(range(n))
    k = _k(full)
    if k is None:
        raise ValueError(
            "expected agreement is 1.0, so kappa is undefined: both raters used a single class. "
            "Reporting 1.0 here would claim reliability the sample cannot demonstrate."
        )
    rng = _random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = _k(idx)
        if v is not None:
            boots.append(v)
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))] if boots else float("nan")
    hi = boots[int(0.975 * (len(boots) - 1))] if boots else float("nan")
    agree = sum(1 for i in full if a[i] == b[i]) / n
    return k, lo, hi, agree


def fleiss_kappa(ratings: list[list[str]]) -> tuple[float, float, float]:
    """Fleiss' kappa for a fixed number of raters per subject (Fleiss, 1971).

    Inputs:  one list per subject holding every rater's category for it. Every list must have the
             same length n >= 2; categories are any hashable labels.
    Outputs: (kappa, mean observed agreement P_bar, chance agreement P_e). kappa is nan when chance
             agreement is 1, i.e. every rating falls in one category, where it is undefined.
    Raises:  ValueError on no subjects, fewer than two raters, or unequal rater counts.
    Complexity: O(subjects x raters).
    """
    if not ratings:
        raise ValueError("no subjects")
    n = len(ratings[0])
    if n < 2 or any(len(r) != n for r in ratings):
        raise ValueError("every subject needs the same number of ratings, at least two")
    cats = sorted({c for r in ratings for c in r}, key=str)
    counts = [[r.count(c) for c in cats] for r in ratings]
    big_n = len(ratings)
    p_bar = sum((sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts) / big_n
    p_e = sum((sum(row[j] for row in counts) / (big_n * n)) ** 2 for j in range(len(cats)))
    kappa = float("nan") if p_e == 1 else (p_bar - p_e) / (1 - p_e)
    return kappa, p_bar, p_e


def stratified_bootstrap_ci(
    strata: Sequence[tuple[int, Sequence]],
    statistic,
    n_boot: int = 5000,
    seed: int = 42,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    """Design-based percentile interval for a statistic of a stratified, without-replacement sample.

    Why this exists rather than a Wilson interval on pooled counts
    --------------------------------------------------------------
    A Wilson interval is exact for one binomial proportion drawn with replacement. The estimates it
    would be applied to here are neither: they are ratios of Horvitz--Thompson expansions over a
    weighted, WITHOUT-replacement stratified design in which each stratum is sampled at its own
    fraction. Pooling the counts discards the weights and the finite-population correction together,
    so the interval would describe a population that was never sampled.

    Method: the finite-population (pseudo-population) bootstrap
    ----------------------------------------------------------
    For each stratum, build a pseudo-population by replicating its `n_h` observations up to size
    `N_h`, then draw `n_h` from it WITHOUT replacement. Recomputing the statistic across replicates
    gives its sampling distribution under the actual design.

    Two properties make this the right choice. The correction is embedded by construction rather than
    bolted on, so a stratum sampled in full (`n_h == N_h`) reproduces itself exactly and contributes
    no variance -- the correct treatment of the unanimous, fully enumerated strata that drove the
    plug-in standard error to zero. And nothing arithmetic is done to the observations, so they may be
    tuples or labels; that is required here because precision and recall depend on a PAIR of
    indicators per row (was it flagged, was it judged positive), which no single number encodes.

    Parameters
    ----------
    strata : sequence of (N_h, observations)
        `N_h` is the stratum population size; `observations` are the sampled units in any form the
        statistic accepts. `n_h` is len(observations).
    statistic : callable
        Maps a list of (N_h, observations) to a scalar, or None where undefined. Undefined replicates
        are skipped rather than counted as zero.

    Returns
    -------
    (point, lo, hi). `lo`/`hi` are NaN when fewer than 2% of replicates were defined, rather than a
    fabricated interval.

    Failure modes
    -------------
    - An empty stratum raises: it has no sampling distribution, and dropping it would change the
      estimand.
    - n_h > N_h raises: the sample and the population table disagree.
    """
    if not strata:
        raise ValueError("no strata supplied")
    for Nh, obs in strata:
        if len(obs) == 0:
            raise ValueError("a stratum has no observations; it has no sampling distribution")
        if len(obs) > Nh:
            raise ValueError(f"n_h={len(obs)} exceeds N_h={Nh}; sample and population table disagree")

    point = statistic([(Nh, list(obs)) for Nh, obs in strata])
    rng = np.random.default_rng(seed)

    # Pseudo-population: each observation is replicated floor(N_h / n_h) times, and the remaining
    # N_h mod n_h copies go to observations chosen at random IN EACH REPLICATE.
    #
    # This was previously `(obs * ceil(N_h / n_h))[:N_h]`, a deterministic trim that gave the FIRST
    # `N_h mod n_h` observations an extra copy for every replicate. That is not a neutral tie-break:
    # in a stratum of N_h=5 from n_h=4 it pinned observation 0 into every pseudo-population, so no
    # replicate could omit it. On this project's data that observation was the single false negative
    # in the whole study, which made the recall interval one-sided -- its upper limit was bit-equal
    # to the point estimate, an artefact of list order rather than of the data.
    base = []
    for Nh, obs in strata:
        obs = list(obs)
        nh = len(obs)
        base.append((Nh, nh, obs, Nh // nh, Nh % nh))

    reps: list[float] = []
    for _ in range(n_boot):
        drawn = []
        for Nh, nh, obs, reps_each, remainder in base:
            pop = obs * reps_each
            if remainder:
                extra = rng.choice(nh, size=remainder, replace=False)
                pop = pop + [obs[j] for j in extra]
            idx = rng.choice(Nh, size=nh, replace=False)
            drawn.append((Nh, [pop[j] for j in idx]))
        v = statistic(drawn)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            reps.append(float(v))
    if len(reps) < 0.02 * n_boot:
        return (point if point is not None else float("nan"), float("nan"), float("nan"))
    a = (1.0 - conf) / 2.0
    return (point if point is not None else float("nan"),
            float(np.quantile(reps, a)), float(np.quantile(reps, 1.0 - a)))
