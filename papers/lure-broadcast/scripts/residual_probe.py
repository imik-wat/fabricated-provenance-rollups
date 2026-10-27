"""Identify the contracts behind the residual `Transfer` rows (reviewer round 2, item 13).

Why the residual bucket needs explaining
----------------------------------------
Token standards are separated here by the width of the event's data field: one 32-byte word for a
fungible transfer, empty for a non-fungible one. Rows matching neither are counted in a residual
bucket rather than forced into a class, and there are 3,024 of them across the corpus. The reviewer
is right that a width test is a proxy for standard semantics and needs defending, and the residual
widths are where the proxy is visible.

The widths are diagnostic rather than arbitrary. `Transfer(address,address,uint256)` hashes to one
topic0 whatever the declaration does with `indexed`, but indexing moves parameters out of the data
field: with both addresses indexed the data holds one word (66 hex characters with the 0x prefix),
with one indexed it holds two (130), and with none it holds three (194). The observed residual is
dominated by 194, which is the unindexed declaration, so those rows are almost certainly fungible
transfers whose participants are not in topics at all. This script tests that reading by finding the
emitting contracts and reporting, for each, the data width, the number of topics present and the
metadata the contract reports.

That distinction matters beyond bookkeeping: a row with no indexed participants cannot be joined on
topic1/topic2, so it is invisible to every analysis in this paper that reads those fields. Counting
such rows as a residual is correct; silently assigning them to either standard would not be.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet  (READ-ONLY)

Outputs
-------
papers/lure-broadcast/results/residual_probe_<chain>.json (+ .manifest.json)

Failure modes
-------------
- A chain whose census records no residual rows is skipped explicitly rather than scanned for
  nothing, and the skip is recorded.
- If the number of residual rows found disagrees with the census, the run RAISES: the two passes
  read the same corpus with the same discriminant and must agree, and a divergence means one of
  them is wrong.

Complexity: one full pass per chain over three columns, parallel by row group.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src.dataio import ERC20_DATA_HEXLEN  # noqa: E402
from src.nonfungible import NONFUNGIBLE_DATA_HEXLEN  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
MAX_WORKERS = 12
#: `topic3` was never selected at extraction time and is absent from the corpus schema, so the topic
#: count below is over topic0..topic2 and saturates at three. That is sufficient for the question
#: asked here, which is whether the participant addresses are indexed at all.
COLUMNS = ["address", "data", "topic0", "topic1", "topic2"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _scan(task):
    path, rg = task
    out = {"rows": 0, "by_contract": Counter(), "by_width": Counter(), "by_shape": Counter(),
           "samples": []}
    tbl = pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS)
    for b in tbl.to_batches(max_chunksize=1_000_000):
        out["rows"] += b.num_rows
        d = b.column("data")
        w = pc.utf8_length(d)
        m = pc.and_(pc.not_equal(w, ERC20_DATA_HEXLEN),
                    pc.not_equal(w, NONFUNGIBLE_DATA_HEXLEN))
        m = pc.fill_null(m, True)          # a null data field is residual, not silently dropped
        if not pc.any(m).as_py():
            continue
        sub = b.filter(m)
        addr = pc.utf8_lower(sub.column("address")).to_pylist()
        wid = pc.utf8_length(sub.column("data")).to_pylist()
        t0 = sub.column("topic0").to_pylist()
        t1 = sub.column("topic1").to_pylist()
        t2 = sub.column("topic2").to_pylist()
        for a, ww, x0, x1, x2 in zip(addr, wid, t0, t1, t2):
            out["by_contract"][a] += 1
            out["by_width"][ww if ww is not None else -1] += 1
            # how many topics are actually present; topic0 is the signature, so a row with only
            # topic0 has no indexed participants at all
            ntop = sum(1 for x in (x0, x1, x2) if x)
            out["by_shape"][f"width={ww},topics={ntop}"] += 1
            if len(out["samples"]) < 5:
                out["samples"].append({"contract": a, "data_hexlen": ww, "topics_present": ntop})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"residual_probe_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    census = json.loads((OUT_ROOT / f"{args.chain}_nf_census.json").read_text())
    expected = census["residual_rows"]
    if expected == 0:
        doc = {"chain": args.chain, "skipped": True,
               "reason": "the census records no residual rows on this chain; nothing to probe"}
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"  {args.chain}: census records 0 residual rows; skipped")
        return 0

    workers = args.workers or max(1, min(MAX_WORKERS, (os.cpu_count() or 4) - 2))
    tasks = []
    for cd in sorted((RAW / args.chain).glob("chunk_*")):
        lp = cd / "logs.parquet"
        if lp.exists() and (cd / "_DONE").exists():
            tasks += [(str(lp), i) for i in range(pq.ParquetFile(lp).num_row_groups)]

    acc = {"rows": 0, "by_contract": Counter(), "by_width": Counter(), "by_shape": Counter(),
           "samples": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_scan, tasks, chunksize=4):
            acc["rows"] += r["rows"]
            for k in ("by_contract", "by_width", "by_shape"):
                acc[k].update(r[k])
            for s in r["samples"]:
                if len(acc["samples"]) < 20:
                    acc["samples"].append(s)
    el = time.time() - t0

    found = sum(acc["by_contract"].values())
    if found != expected:
        raise SystemExit(f"residual rows found ({found}) disagree with the census ({expected}); the "
                         f"two passes use the same discriminant over the same corpus and must agree")

    doc = {
        "chain": args.chain,
        "residual_rows": found,
        "agrees_with_census": True,
        "rows_scanned": acc["rows"],
        "wall_seconds": round(el, 1),
        "distinct_contracts": len(acc["by_contract"]),
        "by_data_hexlen": dict(sorted(acc["by_width"].items())),
        "by_shape": dict(sorted(acc["by_shape"].items(), key=lambda kv: -kv[1])),
        "top_contracts": [{"address": "0x" + a if not a.startswith("0x") else a, "rows": n}
                          for a, n in acc["by_contract"].most_common(15)],
        "samples": acc["samples"],
        "interpretation": ("data width follows the `indexed` declaration: both participants indexed "
                           "gives one data word (66 hex chars), one indexed gives two (130), none "
                           "gives three (194). Rows with fewer participant topics cannot be joined "
                           "on topic1/topic2 and are therefore invisible to the rest of this paper, "
                           "which is why they are counted separately rather than assigned a class."),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"residual_probe_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/residual_probe.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    print(f"  {args.chain}: {found:,} residual rows from {len(acc['by_contract'])} contracts "
          f"in {el/60:.1f} min")
    print(f"    shapes: {dict(list(doc['by_shape'].items())[:6])}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
