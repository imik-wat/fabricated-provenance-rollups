"""A probability sample of the contracts BELOW the detector frame, per chain (IEEE R2.2, R1.2; WP4.1).

The detector population is each chain's volume head: the top 1,200 contracts by transfer count,
registry tokens excluded. Every contract and campaign count is therefore a lower bound, and Reviewer 2
asked for the cost of that selection to be quantified. Table 4's depth sweep shows the counts still
rising at the cutoff, but it cannot say how much lies beyond. This script answers that by sampling.

Design, declared here before any tail metadata is read:
  POPULATION  non-registry contracts in <chain>_token_volumes.csv at rank index >= FRAME_DEPTH,
              excluding every contract in the frame.
  STRATA      rank-index bands TAIL_BANDS, which are log-spaced because volume falls by orders of
              magnitude with rank. A band past the end of a chain's list is dropped.
  SAMPLE      simple random sample without replacement of PER_STRATUM indices per band, seed SEED.
  MEASURE     on-chain name/symbol (latest state), classified by the frozen detector.
  ESTIMATES   per band and in total:
                * the observed flagged count, a hard lower bound;
                * the Horvitz-Thompson point estimate sum_h N_h x_h / n_h;
                * a 95% upper bound: an exact hypergeometric bound per band at alpha/H, summed
                  (Bonferroni), which is conservative. The joint bound of Theorem 1
                  (stats.stratified_exact_upper_bound) enumerates every K_h up to N_h - n_h + x_h.
                  That is feasible for the detector's strata of tens of contracts, but not here,
                  where one band holds over 23 million contracts on Base. Unresolved sampled
                  contracts are treated as UNSAMPLED, which is also conservative, since any of them
                  could be flagged;
                * the HT estimate of the flagged contracts' transfer volume against the tail's total.
              Flagged tail contracts are listed with their campaign key, so it can be read which
              campaigns are new and which are already in the frame.
Metadata is fetched with endpoint rotation, and the shared metadata cache is NOT written: the
cache-save function is swapped out for the duration of the call and restored in a finally block.

Output: papers/lure-broadcast/results/tail_sample_<chain>.json (+ .manifest.json); _PARTIAL.json
        when metadata coverage is below MIN_COVERAGE.
Complexity: one streaming read of the rank file, plus 2 eth_calls per sampled contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, registry, token_meta  # noqa: E402
from build_annotation_sample_v2 import is_unresolved, metadata_path, rank_path  # noqa: E402

FRAME_DEPTH = 1200
TAIL_BANDS = [(1_200, 5_000), (5_000, 20_000), (20_000, 100_000), (100_000, 1_000_000),
              (1_000_000, 10**12)]
PER_STRATUM = 300
SEED = 20261027
MIN_COVERAGE = 0.80
OUT_ROOT = PAPER / "results"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def band_of(index: int, bands=TAIL_BANDS) -> str | None:
    for lo, hi in bands:
        if lo <= index < hi:
            return f"{lo}-{hi}"
    return None


def read_tail(path: Path, excluded: set[str], bands=TAIL_BANDS) -> dict[str, list[tuple[int, str, int]]]:
    """Every tail contract per band as (rank index, address, transfer count), exclusions removed."""
    out: dict[str, list[tuple[int, str, int]]] = {}
    with path.open() as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            b = band_of(i, bands)
            if b is None:
                continue
            a = row["token"].lower()
            if a in excluded:
                continue
            out.setdefault(b, []).append((i, a, int(row["count"])))
    return out


def draw(tail: dict[str, list], per_stratum: int, seed: int) -> dict[str, list[tuple[int, str, int]]]:
    rng = random.Random(seed)
    return {b: sorted(rng.sample(rows, min(per_stratum, len(rows))))
            for b, rows in sorted(tail.items())}


def fetch_without_caching(contracts: list[str], chain: str) -> dict[str, dict]:
    """token_meta.fetch_metadata with endpoint rotation, but without writing the shared cache."""
    saved = token_meta._metadata_cache_save
    try:
        token_meta._metadata_cache_save = lambda *_a, **_k: None
        return token_meta.fetch_metadata(contracts, rpc_url=token_meta.rpc_for(chain), chain=chain)
    finally:
        token_meta._metadata_cache_save = saved


def band_upper(big_n: int, n: int, x: int, alpha: float) -> int:
    """Exact one-sided upper bound on the flagged count K in a band of N, from x flagged among n drawn.

    It is the largest K in [x, N - n + x] with P_K(X <= x) > alpha, X ~ Hyp(N, K, n). That tail is
    non-increasing in K, so a binary search finds it in O(log N) evaluations. With n = 0 nothing was
    learned and the bound is N.
    """
    from scipy.stats import hypergeom
    if n == 0:
        return big_n
    lo, hi = x, big_n - n + x          # lo is always feasible: P_x(X <= x) = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, big_n, mid, n) > alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def estimate(tail: dict[str, list], sample: dict[str, list], verdicts: dict[str, str | None],
             alpha: float = 0.05) -> dict:
    """Per-band and total estimates. verdicts maps address -> strength, or None if unresolved."""
    per, design = {}, []
    ht = ht_vol = 0.0
    observed = 0
    for b in sorted(tail):
        big_n = len(tail[b])
        drawn = sample[b]
        resolved = [(i, a, n) for i, a, n in drawn if verdicts.get(a) is not None]
        flagged = [(i, a, n) for i, a, n in resolved if verdicts[a] == "strong"]
        n_res = len(resolved)
        w = big_n / n_res if n_res else 0.0
        ht += w * len(flagged)
        ht_vol += w * sum(n for _, _, n in flagged)
        observed += len(flagged)
        design.append((big_n, n_res, len(flagged)))
        per[b] = {"N": big_n, "drawn": len(drawn), "resolved": n_res, "strong": len(flagged),
                  "weak": sum(1 for _, a, _ in resolved if verdicts[a] == "weak"),
                  "tail_transfers": sum(n for _, _, n in tail[b]),
                  "ht_strong": w * len(flagged) if n_res else None}
    h = max(len(design), 1)
    uppers = [band_upper(big_n, n, x, alpha / h) for big_n, n, x in design]
    for b, u in zip(sorted(tail), uppers):
        per[b]["upper_strong_bonferroni"] = u
    total_vol = sum(p["tail_transfers"] for p in per.values())
    return {"per_band": per, "observed_strong": observed, "ht_strong": ht,
            "upper95_strong": sum(uppers), "upper95_method": "per-band exact hypergeometric at alpha/H, summed (Bonferroni)",
            "tail_contracts": sum(p["N"] for p in per.values()),
            "ht_strong_transfers": ht_vol, "tail_transfers": total_vol,
            "ht_strong_transfer_share": ht_vol / total_vol if total_vol else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = OUT_ROOT / f"tail_sample_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    raw = json.loads(metadata_path(args.chain).read_text())
    frame = {a.lower() for a in (raw[args.chain] if args.chain == "base" else raw)}
    reg = {"0x" + a.lower() for a in registry.load_token_registry(args.chain).legit_addresses}
    tail = read_tail(rank_path(args.chain), frame | reg)
    sample = draw(tail, args.per_stratum, SEED)
    contracts = sorted({a for rows in sample.values() for _, a, _ in rows})
    print(f"  {args.chain}: tail bands {[(b, len(r)) for b, r in sorted(tail.items())]}; "
          f"sampling {len(contracts)}")
    md = fetch_without_caching(contracts, args.chain)
    verdicts, keys = {}, {}
    for a in contracts:
        e = md.get(a) or {}
        if is_unresolved(e):
            verdicts[a] = None
            continue
        v = ml.classify(e.get("name"), e.get("symbol"))
        verdicts[a] = v.strength
        if v.strength == "strong":
            keys[a] = ml.campaign_key(v) or ""
    est = estimate(tail, sample, verdicts)
    frame_keys = set()
    fm = raw[args.chain] if args.chain == "base" else raw
    for e in fm.values():
        if isinstance(e, dict) and not is_unresolved(e):
            v = ml.classify(e.get("name"), e.get("symbol"))
            if v.strength == "strong":
                frame_keys.add(ml.campaign_key(v) or "")
    flagged_rows = [{"address": a, "rank_index": i, "transfers": n, "band": b,
                     "name": (md.get(a) or {}).get("name", ""), "symbol": (md.get(a) or {}).get("symbol", ""),
                     "campaign_key": keys[a], "campaign_in_frame": keys[a] in frame_keys}
                    for b, rows in sample.items() for i, a, n in rows if a in keys]
    coverage = sum(1 for v in verdicts.values() if v is not None) / len(verdicts) if verdicts else 0.0
    doc = {"chain": args.chain, "seed": SEED, "per_stratum": args.per_stratum, "frame_depth": FRAME_DEPTH,
           "bands": [f"{lo}-{hi}" for lo, hi in TAIL_BANDS], "metadata_coverage": coverage,
           "unresolved_treated_as_unsampled": True, **est,
           "strength_counts": dict(Counter(v for v in verdicts.values() if v is not None)),
           "flagged_in_sample": flagged_rows,
           "new_campaign_keys": sorted({r["campaign_key"] for r in flagged_rows if not r["campaign_in_frame"]})}
    target = out if coverage >= MIN_COVERAGE else out.with_name(out.stem + "_PARTIAL.json")
    target.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    target.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/tail_sample.py", "git_commit": git_commit(),
        "seed": SEED}, indent=1) + "\n")
    print(f"  coverage {coverage:.3f}; strong observed {est['observed_strong']}, HT {est['ht_strong']:.1f}, "
          f"upper95 {est['upper95_strong']} of {est['tail_contracts']:,} tail contracts; "
          f"HT flagged transfer share of tail {est['ht_strong_transfer_share']}")
    print(f"  -> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
