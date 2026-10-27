"""Per-contract ERC-20 transfer census for a chain -> <chain>_token_volumes.csv.

Why this exists
---------------
Four of the five chains have a cached per-contract transfer census in the shared `results/` tree;
Polygon zkEVM does not, so the fifth chain is absent from every volume-tier figure (defect 4 in
docs/DEFECTS.md). That census is the sampling frame for this paper's rank-stratified prevalence
estimator, so the gap has to be closed before any stratified number can be computed.

This deliberately re-runs the existing, tested primitive (`dataio.token_counts`) rather than
reimplementing it, so the new chain's numbers are produced by exactly the same code path as the four
that already exist and are directly comparable to them.

Isolation
---------
Writes ONLY under papers/lure-broadcast/results/. It never writes to the shared `results/` tree,
whose artefacts are pinned by scripts/baseline_guard.py -- including for chains that already have a
census there, so a re-run can never overwrite one.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet   (read-only)

Outputs
-------
papers/lure-broadcast/results/<chain>_token_volumes.csv   (token,count -- descending by count)
papers/lure-broadcast/results/<chain>_token_volumes.manifest.json

Failure modes
-------------
- An unknown chain, or a chain directory that is absent/empty, raises rather than writing an empty
  census that would read as "this chain has no tokens".
- Refuses to overwrite an existing output without --force.
- `require_done=True` (the dataio default) means chunks without a _DONE marker are skipped, so a
  partial pull cannot silently produce a partial census; the chunk count is recorded in the manifest
  so the coverage is auditable.

Complexity: O(rows) time, O(distinct contracts) memory -- streamed row groups, two columns only.
Measured: Polygon zkEVM ~1.5 s / 8,233 contracts; Base is ~53 s per 5M-block chunk.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import dataio  # noqa: E402  (path set above so the shared library is importable)

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]


def git_commit() -> str:
    """Short HEAD, for provenance. Never fatal: provenance must not block a measurement."""
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chain", choices=CHAINS)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    chain_dir = RAW / args.chain
    if not chain_dir.is_dir():
        raise SystemExit(f"no chain directory at {chain_dir} -- refusing to write an empty census")
    chunks = dataio.chunk_dirs(chain_dir)
    done = [c for c in chunks if (c / "_DONE").exists()]
    if not done:
        raise SystemExit(f"{args.chain}: {len(chunks)} chunk dirs but none carry a _DONE marker; "
                         f"refusing to publish a census of an incomplete pull")

    out_csv = OUT_ROOT / f"{args.chain}_token_volumes.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"{out_csv.relative_to(ROOT)} exists; pass --force to overwrite")

    t0 = time.time()
    counts = dataio.token_counts(chain_dir, args.chain)
    elapsed = time.time() - t0
    if not counts:
        raise SystemExit(f"{args.chain}: token_counts returned nothing over {len(done)} chunks -- "
                         f"this is a failure, not an empty chain")

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "count"])
        w.writerows(ordered)

    total = sum(counts.values())
    manifest = {
        "output": out_csv.name,
        "chain": args.chain,
        "input": str(chain_dir.relative_to(ROOT)),
        "chunks_total": len(chunks),
        "chunks_with_done": len(done),
        "distinct_contracts": len(counts),
        "erc20_transfer_rows": total,
        "wall_seconds": round(elapsed, 2),
        "git_commit": git_commit(),
        "producer": "papers/lure-broadcast/scripts/token_volumes.py",
        "primitive": "analysis/src/dataio.py:token_counts",
    }
    (OUT_ROOT / f"{args.chain}_token_volumes.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")

    print(f"{args.chain}: {len(counts):,} contracts, {total:,} ERC-20 transfer rows, "
          f"{len(done)}/{len(chunks)} chunks, {elapsed:.1f} s")
    print(f"  -> {out_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
