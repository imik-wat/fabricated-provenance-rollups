"""Census the metadata-lure channel: every transfer from every lure contract, whole chain.

Why a full pass rather than the sampled frame
---------------------------------------------
The sampled sub-corpus was right for the overlap tests, where both arms share the sampling frame and the
statistic is a ratio. It is wrong for three things this script produces:

  1. CAMPAIGN COMPLETENESS. At 40 row groups per chunk only 30 of Base's 48 campaigns appeared at all. A
     campaign census cannot be built from a frame that misses 38% of the campaigns.
  2. FAN-OUT AND DISTINCT RECIPIENTS. These are absolute quantities, and a sample under-counts them by an
     unknown factor -- the exact error that invalidated the discarded "0 of 3 operators" figure.
  3. LIFESPAN. First and last block seen are order statistics; a sample truncates both ends inward.

So this reads every row of the chain and keeps only the lure contracts' transfers.

Memory
------
Distinct recipients across the channel run to tens of millions. Addresses are stored as 20-byte `bytes`
rather than 42-character hex strings: ~53 bytes per set entry instead of ~115, which brings 50M recipients
from roughly 6 GB down to under 3 GB. Hex is restored only on output.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet   (read-only, ALL row groups)
analysis/config/token_metadata.json

Outputs
-------
papers/lure-broadcast/results/campaign_census_<chain>.json (+ .manifest.json)

Failure modes
-------------
- Zero matched rows RAISES rather than emitting an empty census.
- A campaign present in the metadata but absent from the chain is reported with zero transfers rather than
  omitted, so the difference between "no activity" and "not looked at" stays visible.
- Fan-out is reported as transfers / distinct recipients and is undefined (null) for a campaign with no
  recipients, never 0.

Complexity: one full pass, three columns projected. Measured ~10M rows/s on this machine, so Base's
10.95B rows take roughly 20-35 minutes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402
from src.dataio import chunk_dirs  # noqa: E402
from src.heuristic import ZERO_ADDRESS  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
ZERO20 = bytes(20)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"campaign_census_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[args.chain]
    campaign_of: dict[str, str] = {}
    for a, m in md.items():
        v = ml.classify(m.get("name"), m.get("symbol"))
        if v.strength == "strong":
            k = ml.campaign_key(v)
            if k:
                campaign_of[a.lower()] = k
    if not campaign_of:
        raise SystemExit("no lure contracts")
    campaigns = sorted(set(campaign_of.values()))
    lure_arr = pa.array(sorted(campaign_of))
    print(f"  {args.chain}: {len(campaign_of)} contracts in {len(campaigns)} campaigns; full-chain pass")

    recips: dict[str, set[bytes]] = defaultdict(set)
    transfers: dict[str, int] = defaultdict(int)
    contracts_seen: dict[str, set[str]] = defaultdict(set)
    senders: dict[str, int] = defaultdict(int)
    first_b: dict[str, int] = {}
    last_b: dict[str, int] = {}
    rows = matched = 0
    t0 = time.time()
    for cd in chunk_dirs(RAW / args.chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        pf = pq.ParquetFile(lp)
        for b in pf.iter_batches(batch_size=1_000_000,
                                 columns=["address", "topic1", "topic2", "block_number"]):
            rows += b.num_rows
            addr = pc.utf8_lower(b.column("address"))
            sub = b.filter(pc.is_in(addr, value_set=lure_arr))
            if not sub.num_rows:
                continue
            matched += sub.num_rows
            a_ = pc.utf8_lower(sub.column("address")).to_pylist()
            f_ = sub.column("topic1").to_pylist()
            r_ = sub.column("topic2").to_pylist()
            bl = sub.column("block_number").to_pylist()
            for ai, fi, ri, bi in zip(a_, f_, r_, bl):
                k = campaign_of[ai]
                transfers[k] += 1
                contracts_seen[k].add(ai)
                if ri:
                    rb = bytes.fromhex(ri[-40:])
                    if rb != ZERO20:
                        recips[k].add(rb)
                if fi:
                    fb = "0x" + fi[-40:]
                    if fb != ZERO_ADDRESS:
                        senders[fb] += 1
                if k not in first_b or bi < first_b[k]:
                    first_b[k] = bi
                if k not in last_b or bi > last_b[k]:
                    last_b[k] = bi
        print(f"    {cd.name}: {rows:,} rows scanned, {matched:,} matched, "
              f"{time.time()-t0:.0f}s", flush=True)
    el = time.time() - t0
    if not matched:
        raise SystemExit("zero lure rows matched -- refusing to emit an empty census")

    per_campaign = {}
    for k in campaigns:                       # every campaign, including absent ones
        n_r = len(recips.get(k, ()))
        n_t = transfers.get(k, 0)
        per_campaign[k] = {
            "contracts_in_metadata": sum(1 for c, kk in campaign_of.items() if kk == k),
            "contracts_observed": len(contracts_seen.get(k, ())),
            "transfers": n_t,
            "distinct_recipients": n_r,
            "fanout_ratio": (n_t / n_r) if n_r else None,
            "first_block": first_b.get(k), "last_block": last_b.get(k),
            "block_span": (last_b[k] - first_b[k]) if k in first_b else None,
        }
    all_recips = set().union(*recips.values()) if recips else set()
    absent = [k for k in campaigns if not transfers.get(k)]
    top_ops = sorted(senders.items(), key=lambda kv: -kv[1])[:50]

    doc = {
        "chain": args.chain,
        "scope": "FULL CHAIN -- every row scanned, lure-contract rows retained. Not a sample.",
        "rows_scanned": rows, "rows_matched": matched, "wall_seconds": round(el, 1),
        "contracts": len(campaign_of), "campaigns": len(campaigns),
        "campaigns_absent_from_chain": absent,
        "channel_totals": {
            "transfers": sum(transfers.values()),
            "distinct_recipients_union": len(all_recips),
            "distinct_operators": len(senders),
            "fanout_ratio_channel": (sum(transfers.values()) / len(all_recips)) if all_recips else None,
        },
        "per_campaign": per_campaign,
        "top_operators_by_volume": [{"address": a, "transfers": n} for a, n in top_ops],
        "note": ("fanout_ratio is transfers per distinct recipient; ~1.0 means every recipient was reached "
                 "once, the broadcast signature. Values far above 1 indicate circulation rather than "
                 "one-shot delivery."),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"campaign_census_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/campaign_census.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    print(f"\n  channel: {sum(transfers.values()):,} transfers to "
          f"{len(all_recips):,} distinct recipients "
          f"(fan-out {doc['channel_totals']['fanout_ratio_channel']:.4f}), "
          f"{len(senders):,} operators")
    print(f"  campaigns with no on-chain activity: {len(absent)}")
    print(f"  largest campaigns by distinct recipients:")
    for k, v in sorted(per_campaign.items(), key=lambda kv: -(kv[1]["distinct_recipients"] or 0))[:8]:
        print(f"    {k:28s} {v['distinct_recipients']:>10,} recipients  "
              f"{v['transfers']:>10,} transfers  fan-out "
              f"{v['fanout_ratio']:.3f}" if v['fanout_ratio'] else "")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
