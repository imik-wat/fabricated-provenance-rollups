"""Census every Transfer log row of a chain by token standard -> the paper's contribution 1.

Why this exists
---------------
`dataio._decode_logs` keeps only 32-byte-`data` rows, so roughly two billion non-fungible rows have never
been counted. This produces that count, per chain, over complete histories, and it also settles a factual
error: `dataio.py`'s docstring and `paper/sections/methodology.tex` both state that a large majority of
`Transfer` logs are ERC-721 (specifically "~76%" on Polygon zkEVM), against a measured 9.325%. See
defect 2 in docs/DEFECTS.md.

It additionally makes the figure `2,020,743,910` traceable for the first time (defect 8): that number
appears nowhere in `results/` or `paper/generated/`, only in a docstring, which violates golden rule 1.

Isolation
---------
Reads `data/raw/hypersync/` strictly read-only and writes ONLY under
papers/lure-broadcast/results/. It never touches the shared `results/` tree, which is pinned by
scripts/baseline_guard.py.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet

Outputs
-------
papers/lure-broadcast/results/<chain>_nf_census.json  (+ .manifest.json)
papers/lure-broadcast/results/nf_census_all.json      (cross-chain roll-up, --rollup)

Failure modes
-------------
- `nonfungible.census` asserts internally that erc20 + nonfungible + residual == total, so a partition
  failure raises rather than publishing a share that silently omits rows.
- A chain with no _DONE chunks raises rather than reporting a zero share.
- Shares are `null`, never 0.0, for an empty chain -- 0.0 would read as a measurement.
- The roll-up refuses to run if any per-chain file is missing, so a partial roll-up cannot be mistaken
  for a five-chain total.

Complexity: O(rows), streamed at ~11.6M rows/s per worker on the four-column projection. Measured:
Polygon zkEVM 2.4 s; the full 19.33B-row corpus is a few hours single-threaded, dominated by Base.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import nonfungible as nf  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
CHAINS = ["polygon-zkevm", "optimism", "arbitrum-one", "zksync-era", "base"]  # cheapest first


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def census_chain(chain: str, force: bool) -> dict:
    out = OUT_ROOT / f"{chain}_nf_census.json"
    if out.exists() and not force:
        print(f"  {chain}: exists, skipping (pass --force to redo)")
        return json.loads(out.read_text())
    chain_dir = RAW / chain
    if not chain_dir.is_dir():
        raise SystemExit(f"no chain directory {chain_dir}")
    t0 = time.time()
    c = nf.census(chain_dir, chain)
    c["wall_seconds"] = round(time.time() - t0, 1)
    if c["chunks_used"] == 0:
        raise SystemExit(f"{chain}: no chunk carries a _DONE marker; refusing to publish a zero share")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"{chain}_nf_census.manifest.json").write_text(json.dumps({
        "output": out.name,
        "producer": "papers/lure-broadcast/scripts/run_nf_census.py",
        "primitive": "analysis/src/nonfungible.py:census",
        "input": str(chain_dir.relative_to(ROOT)),
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    share = c["nonfungible_share"]
    print(f"  {chain:14s} total {c['total_rows']:>14,}  nf {c['nonfungible_rows']:>13,} "
          f"({share*100:6.3f}%)  residual {c['residual_rows']:>10,}  {c['wall_seconds']:7.1f}s")
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chains", nargs="*", default=None,
                    help=f"chains to census (default all, cheapest first: {' '.join(CHAINS)})")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rollup", action="store_true", help="write the cross-chain roll-up")
    args = ap.parse_args()

    targets = args.chains or CHAINS
    for c in targets:
        if c not in CHAINS:
            raise SystemExit(f"unknown chain {c}; known: {CHAINS}")

    results = {c: census_chain(c, args.force) for c in targets}

    if args.rollup:
        missing = [c for c in CHAINS if not (OUT_ROOT / f"{c}_nf_census.json").exists()]
        if missing:
            raise SystemExit(f"refusing to write a roll-up while {missing} are missing -- a partial "
                             f"total would be mistaken for a five-chain figure")
        allc = {c: json.loads((OUT_ROOT / f"{c}_nf_census.json").read_text()) for c in CHAINS}
        tot = sum(v["total_rows"] for v in allc.values())
        nfr = sum(v["nonfungible_rows"] for v in allc.values())
        e20 = sum(v["erc20_rows"] for v in allc.values())
        res = sum(v["residual_rows"] for v in allc.values())
        roll = {
            "chains": CHAINS,
            "total_rows": tot, "erc20_rows": e20, "nonfungible_rows": nfr, "residual_rows": res,
            "nonfungible_share": nfr / tot if tot else None,
            "per_chain_share": {c: allc[c]["nonfungible_share"] for c in CHAINS},
            "per_chain_mint_share": {c: allc[c]["mint_share_of_nonfungible"] for c in CHAINS},
            "erc1155_covered": False,
            "corrects": ("dataio.py docstring and paper/sections/methodology.tex assert a large majority "
                         "of Transfer logs are ERC-721 (~76% on Polygon zkEVM). Measured here."),
            "supersedes_untraceable_figure": 2020743910,
        }
        assert e20 + nfr + res == tot, "roll-up does not partition"
        (OUT_ROOT / "nf_census_all.json").write_text(json.dumps(roll, indent=1, sort_keys=True) + "\n")
        print(f"\n  ROLL-UP  total {tot:,}  erc20 {e20:,}  nonfungible {nfr:,} "
              f"({roll['nonfungible_share']*100:.3f}%)  residual {res:,}")
        print(f"  -> {(OUT_ROOT / 'nf_census_all.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
