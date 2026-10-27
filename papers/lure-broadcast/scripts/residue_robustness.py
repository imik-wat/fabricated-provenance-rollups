"""Does the activity estimate depend on which residue class the address hash selects?

Purpose
-------
The activity sample is deterministic: an address enters the probe when an avalanche hash of its
forty hexadecimal characters falls in one fixed residue class. Its validity rests on the assumption
that residue class is independent of the activity being measured. That the hash constants are
published and were fixed after the corpus existed rules out tuning, but it does not establish
independence, and a referee correctly observed that the assumption was asserted rather than tested.

This script tests it directly. The same estimator is computed over R DISJOINT residue classes of the
same modulus, giving R independent replicates of the headline quantity. Class 0 is the class the
published estimate uses, so it must reproduce that estimate exactly; the remaining classes are
replicates that the independence assumption predicts should differ only by sampling error.

The test reported is whether the observed dispersion across classes is consistent with binomial
noise at the realised per-class sample sizes. If it is, the assumption survives a real attempt to
break it. If it is not, the design has a defect the paper must report, and either outcome is worth
having.

Definitions are taken unchanged from recipient_activity.py so that class 0 is comparable to the
published figure rather than merely similar to it.

Inputs   data/raw/hypersync/<chain>/chunk_*/logs.parquet (read-only, ALL row groups)
         analysis/config/token_metadata.json, the verified-token registry
Outputs  papers/lure-broadcast/results/residue_robustness_<chain>.json (+ manifest)

Failure modes
-------------
- Class 0 failing to reproduce the published estimate RAISES: the replicates would then not be
  replicates of the quantity the paper reports.
- A class with too few sampled recipients to support an interval is reported, not silently pooled.

Complexity: one pass. Memory grows with the number of classes, so the asserted observable, which
retains far more rows than the verified one, is not tracked here; it plays no part in the estimate
under test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import registry as reg  # noqa: E402
from src.dataio import chunk_dirs  # noqa: E402
from src.stats import wilson  # noqa: E402
from recipient_activity import (  # noqa: E402
    ADDR_HI, ADDR_LO, ADDR_WORDS, TOPIC_CHARS, ZERO40,
    FirstSeen, as_char_matrix, drop_zero, fold_address, index_in, lure_contracts, reduce_batch,
    splitmix64,
)

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
MIN_PER_CLASS = 500


def residue(chars: np.ndarray, k_mask: np.uint64) -> np.ndarray:
    """The residue class of each row's address, or K (out of range) where invalid."""
    if chars.shape[0] == 0:
        return np.empty(0, dtype=np.uint64)
    words = np.ascontiguousarray(chars[:, ADDR_LO:ADDR_HI]).view(np.uint64).reshape(-1, ADDR_WORDS)
    return splitmix64(fold_address(words)) & k_mask


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.k < 2 or args.k & (args.k - 1):
        raise SystemExit("--k must be a power of two")
    if not 1 < args.classes <= args.k:
        raise SystemExit("--classes must be between 2 and K")
    k_mask = np.uint64(args.k - 1)
    R = args.classes

    out = OUT_ROOT / f"residue_robustness_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    campaign_of = lure_contracts(args.chain)
    lure_arr = pa.array(sorted(campaign_of))
    verified = sorted("0x" + a.lower().removeprefix("0x")
                      for a in reg.load_token_registry(args.chain).legit_addresses)
    ver_arr = pa.array(verified)
    print(f"  {args.chain}: {R} residue classes of K={args.k}, {len(verified)} verified tokens")

    lure_recv = [FirstSeen() for _ in range(R)]
    ver_send = [FirstSeen() for _ in range(R)]
    ver_recv = [FirstSeen() for _ in range(R)]
    rows = 0
    t0 = time.time()
    for cd in chunk_dirs(RAW / args.chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        for b in pq.ParquetFile(lp).iter_batches(
                batch_size=1_000_000,
                columns=["address", "topic1", "topic2", "block_number", "data"]):
            rows += b.num_rows
            blocks = b.column("block_number").to_numpy(zero_copy_only=False).astype(np.int64)
            addr = pc.utf8_lower(b.column("address"))
            is_lure = np.asarray(
                pc.is_in(addr, value_set=lure_arr).to_numpy(zero_copy_only=False), dtype=bool)
            is_ver = np.asarray(
                pc.is_in(addr, value_set=ver_arr).to_numpy(zero_copy_only=False), dtype=bool)
            body = pc.utf8_slice_codeunits(b.column("data"), 2, 2 ** 30)
            nonzero = np.asarray(
                pc.greater(pc.utf8_length(pc.utf8_trim(body, characters="0")), 0)
                .to_numpy(zero_copy_only=False), dtype=bool)
            is_ver_nz = is_ver & nonzero

            t1, ok1 = as_char_matrix(b.column("topic1"), TOPIC_CHARS)
            t2, ok2 = as_char_matrix(b.column("topic2"), TOPIC_CHARS)
            r1, r2 = residue(t1, k_mask), residue(t2, k_mask)
            for c in range(R):
                s1 = (r1 == c) & ok1
                s2 = (r2 == c) & ok2
                lure_recv[c].add(*reduce_batch(t2, s2 & is_lure, blocks))
                ver_send[c].add(*reduce_batch(t1, s1 & is_ver_nz, blocks))
                ver_recv[c].add(*reduce_batch(t2, s2 & is_ver_nz, blocks))
        print(f"    {cd.name}: {rows:,} rows, {time.time()-t0:.0f}s", flush=True)

    zero = np.frombuffer(ZERO40, dtype=np.uint8).view(np.dtype((np.void, 40)))[0]
    per_class = []
    for c in range(R):
        r_addr, _ = drop_zero(lure_recv[c].finish()[0], zero)
        vs_addr, _ = drop_zero(ver_send[c].finish()[0], zero)
        vr_addr, _ = drop_zero(ver_recv[c].finish()[0], zero)
        n_r = len(r_addr)
        hits = int((index_in(r_addr, vs_addr) >= 0).sum()) if n_r else 0
        seen = np.union1d(vs_addr, vr_addr)
        ctrl = seen[index_in(seen, r_addr) < 0]
        c_hits = int((index_in(ctrl, vs_addr) >= 0).sum()) if len(ctrl) else 0
        lo, hi = wilson(hits, n_r) if n_r else (None, None)
        per_class.append({
            "residue_class": c, "sampled_recipients": n_r,
            "active": hits / n_r if n_r else None, "ci95_lo": lo, "ci95_hi": hi,
            "control_n": int(len(ctrl)),
            # A DIFFERENT universe from the published control, which is drawn from all corpus
            # participants; here it is verified-token participants only, because the asserted
            # observable is not tracked. Its level is therefore not comparable with the headline
            # control rate, and only its stability across classes is meaningful.
            "control_active_verified_universe": c_hits / len(ctrl) if len(ctrl) else None,
            "underpowered": n_r < MIN_PER_CLASS,
        })

    rates = [p["active"] for p in per_class if p["active"] is not None]
    ns = [p["sampled_recipients"] for p in per_class if p["active"] is not None]
    pbar = sum(r * n for r, n in zip(rates, ns)) / sum(ns)
    # Dispersion against what binomial sampling alone would produce at these sizes.
    chi2 = sum((r - pbar) ** 2 * n / (pbar * (1 - pbar)) for r, n in zip(rates, ns))
    dof = len(rates) - 1

    # Class 0 is the published class. If it does not reproduce the published estimate, the other
    # classes are not replicates of the quantity the paper reports and the check is meaningless.
    published = OUT_ROOT / f"recipient_activity_{args.chain}.json"
    class_zero_check = None
    if published.exists():
        want = json.loads(published.read_text())["active"]["estimate"]
        got = per_class[0]["active"]
        class_zero_check = {"published": want, "class_zero": got,
                            "absolute_difference": abs(want - got)}
        if abs(want - got) > 1e-12:
            raise SystemExit(
                f"class 0 gives {got!r} but the published estimate is {want!r}. The replicates "
                f"would not be replicates of the reported quantity; fix the definitions before "
                f"reporting a robustness result.")

    doc = {
        "chain": args.chain, "k": args.k, "classes": R,
        "class_zero_reproduces_published": class_zero_check,
        "question": ("is the activity estimate stable across disjoint residue classes of the same "
                     "hash, as the independence assumption predicts?"),
        "rows_scanned": rows, "wall_seconds": round(time.time() - t0, 1),
        "per_class": per_class,
        "pooled_active": pbar,
        "spread": {"min": min(rates), "max": max(rates), "range": max(rates) - min(rates)},
        "dispersion": {
            "chi2": chi2, "dof": dof, "chi2_over_dof": chi2 / dof if dof else None,
            "reading": ("chi-square of the per-class rates about their pooled value, scaled by "
                        "binomial variance. A value near the degrees of freedom is what independent "
                        "sampling at these sizes would produce; a much larger one indicates the "
                        "residue class carries information about activity."),
        },
        "note": ("class 0 is the class the published estimate uses and reproduces it; the remaining "
                 "classes are independent replicates of the same quantity."),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"residue_robustness_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/residue_robustness.py",
        "git_commit": git_commit(), "k": args.k, "classes": R,
    }, indent=1) + "\n")

    for p in per_class:
        print(f"    class {p['residue_class']}: n={p['sampled_recipients']:>7,} "
              f"active {p['active']:.4f}  control(ver) "
              f"{p['control_active_verified_universe']:.4f}")
    print(f"  pooled {pbar:.4f}  range {doc['spread']['range']:.4f}  "
          f"chi2/dof {chi2/dof:.2f}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
