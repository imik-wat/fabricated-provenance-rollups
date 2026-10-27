"""Intervals for the tail sample's estimates of flagged contracts and flagged transfer volume (WP4.1).

tail_sample.py reports Horvitz-Thompson point estimates and an exact count bound, but no interval for
the flagged TRANSFER VOLUME in the tail. Volume is the quantity that says how much of the channel the
frame misses. This script adds that interval from the saved result, with no further RPC: in each band
the sampled contracts are a simple random sample, a flagged contract contributes its transfer count
and an unflagged one zero, and the HT total sum_h N_h * mean_h is bootstrapped by the paper's
within-band bootstrap: each band's sample is resampled with replacement, and the replicates are
vectorised. The paper's finite-population bootstrap (stats.stratified_bootstrap_ci) builds a
pseudo-population of size N_h in every replicate, which is infeasible when one band holds 23.6
million contracts. At sampling fractions this small, the finite-population correction it would apply
is negligible, and omitting it can only widen the interval. The count gets the same treatment, as a
cross-check on the conservative Bonferroni bound.

Caveat recorded in the output: flagged contracts are rare, and their transfer counts are heavy-tailed
and clustered by campaign (one Base band is dominated by clones of a single campaign). A percentile
bootstrap can under-cover in that regime, so the interval is reported as indicative, beside the exact
count bound, never instead of it.

The frame's own flagged volume is read from forgery_chain_<chain>.json, so the tail estimate can be
stated relative to it.

Output: papers/lure-broadcast/results/tail_volume_<chain>.json (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np

N_BOOT = 5000


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def band_observations(doc: dict) -> list[tuple[int, list[int]]]:
    """Per band: (N_h, per-sampled-contract flagged transfers), with zeros for unflagged resolved."""
    flagged: dict[str, list[int]] = {}
    for r in doc["flagged_in_sample"]:
        flagged.setdefault(r["band"], []).append(int(r["transfers"]))
    out = []
    for band, p in sorted(doc["per_band"].items()):
        vals = flagged.get(band, [])
        if len(vals) != p["strong"]:
            raise ValueError(f"band {band}: {len(vals)} flagged rows listed but {p['strong']} counted")
        if p["resolved"]:
            out.append((p["N"], vals + [0] * (p["resolved"] - len(vals))))
    return out


def ht_total(strata) -> float:
    return sum(n_h * (sum(obs) / len(obs)) for n_h, obs in strata if obs)


def bootstrap_total(strata, n_boot: int = N_BOOT, seed: int = 42) -> tuple[float, float, float]:
    """HT total with a percentile interval from a within-band with-replacement bootstrap."""
    rng = np.random.default_rng(seed)
    point = ht_total(strata)
    reps = np.zeros(n_boot)
    for n_h, obs in strata:
        arr = np.asarray(obs, dtype=float)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        reps += n_h * arr[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def worst_case_volume(tail: dict[str, list], per_band: dict[str, dict]) -> int:
    """Deterministic upper bound on flagged tail transfers, valid whatever the sample showed.

    Per band, at most U_h contracts are flagged (the exact Bonferroni count bound). Those can carry at
    most the U_h largest transfer counts in the band. tail_sample.read_tail returns each band's
    contracts in rank order, i.e. by descending count, so they are the first U_h. This bound is not
    an estimate: it is what the exact count bound implies for volume.
    """
    total = 0
    for band, rows in tail.items():
        u = int(per_band[band]["upper_strong_bonferroni"])
        total += sum(n for _, _, n in rows[:u])
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = PAPER / "results" / f"tail_volume_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")
    doc = json.loads((PAPER / "results" / f"tail_sample_{args.chain}.json").read_text())
    vol = band_observations(doc)
    cnt = [(n, [1 if v > 0 else 0 for v in obs]) for n, obs in vol]
    v_pt, v_lo, v_hi = bootstrap_total(vol)
    c_pt, c_lo, c_hi = bootstrap_total(cnt)
    frame = json.loads((PAPER / "results" / f"forgery_chain_{args.chain}.json").read_text())
    frame_vol = sum(int(n) for n in frame["lure_transfers_by_contract"].values())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tail_sample as ts
    from build_annotation_sample_v2 import metadata_path, rank_path
    from src import registry
    raw = json.loads(metadata_path(args.chain).read_text())
    frame_set = {a.lower() for a in (raw[args.chain] if args.chain == "base" else raw)}
    reg = {"0x" + a.lower() for a in registry.load_token_registry(args.chain).legit_addresses}
    worst = worst_case_volume(ts.read_tail(rank_path(args.chain), frame_set | reg), doc["per_band"])
    res = {"chain": args.chain, "n_boot": N_BOOT,
           "ht_flagged_transfers": {"point": v_pt, "ci95": [v_lo, v_hi]},
           "ht_flagged_contracts": {"point": c_pt, "ci95": [c_lo, c_hi],
                                    "exact_bonferroni_upper95": doc["upper95_strong"]},
           "frame_flagged_transfers": frame_vol,
           "worst_case_flagged_tail_transfers": worst,
           "worst_case_tail_to_frame_ratio": worst / frame_vol if frame_vol else None,
           "tail_to_frame_volume_ratio": {"point": v_pt / frame_vol if frame_vol else None,
                                          "ci95": [v_lo / frame_vol, v_hi / frame_vol] if frame_vol else None},
           "caveat": ("rare, heavy-tailed and campaign-clustered flagged volumes; percentile bootstrap "
                      "intervals are indicative and may under-cover; the exact count bound stands beside them")}
    out.write_text(json.dumps(res, indent=1) + "\n")
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/tail_volume_ci.py", "git_commit": git_commit(),
        "source": f"results/tail_sample_{args.chain}.json"}, indent=1) + "\n")
    print(f"  {args.chain}: tail flagged transfers HT {v_pt:,.0f} [{v_lo:,.0f}, {v_hi:,.0f}] vs frame {frame_vol:,}"
          f"  (ratio {res['tail_to_frame_volume_ratio']['point']}); contracts HT {c_pt:,.0f} [{c_lo:,.0f}, {c_hi:,.0f}];"
          f" worst-case tail volume {worst:,} (x{res['worst_case_tail_to_frame_ratio']:.3f} frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
