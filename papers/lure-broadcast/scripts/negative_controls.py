"""Negative controls: legitimate NFT collections and registry-listed tokens (IEEE review R2.4; WP6).

Reviewer 2 asked whether the current rules could produce false positives on legitimate token
distributions and NFT applications. Both control sets are chosen independently of the detector:
  * NFT: collections on CoinGecko's curated list for the chain (pinned snapshot, fetch_nft_list.py);
  * FUNGIBLE: the registry-listed tokens, identified by CoinGecko reverse lookup (registry.py).
Each control contract gets the same three readings as the channel:
  1. the frozen metadata rule on its on-chain name and symbol (fetched now, never written to the
     shared metadata cache);
  2. its addressing pattern over the full corpus: transfers, distinct recipients, fan-out phi and
     the share of transfers that are mints;
  3. the ledger and authorisation probe of forgery_probe.py, pinned to the corpus end block.
The point to establish is which readings DISCRIMINATE. A legitimate mint or airdrop also names each
recipient about once, so a fan-out near 1 is expected among the controls and is reported as such,
not hidden. The metadata conjunction and the contract-state probe are the readings that can
separate them.

Output: papers/lure-broadcast/results/negative_controls_<chain>.json (+ .manifest.json)
Failure modes: no controls at all raises. Metadata coverage below MIN_COVERAGE writes
_PARTIAL.json. No archive endpoint raises, as in forgery_probe.
Complexity: one full-chain pass (minutes) plus a few RPC batches.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, registry, token_meta  # noqa: E402
from src.dataio import chunk_dirs  # noqa: E402
import forgery_probe as fp  # noqa: E402
import rule_paths as rp  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
SNAPSHOT = PAPER / "data" / "negative_controls" / "coingecko_nfts.json"
OUT_ROOT = PAPER / "results"
#: CoinGecko asset_platform_id for each chain. zkSync Era and Polygon zkEVM have no collection on
#: the pinned list, so their NFT arm is empty and is reported as empty.
PLATFORM = {"base": "base", "arbitrum-one": "arbitrum-one", "optimism": "optimistic-ethereum",
            "zksync-era": "zksync", "polygon-zkevm": "polygon-zkevm"}
MIN_COVERAGE = 0.80
ZERO20 = bytes(20)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def nft_controls(snapshot: list[dict], chain: str) -> list[str]:
    plat = PLATFORM[chain]
    return sorted({r["contract_address"].lower() for r in snapshot
                   if r.get("asset_platform_id") == plat and r.get("contract_address")})


def addressing(chain: str, contracts: list[str]) -> dict[str, dict]:
    """Per contract over every row of the chain: transfers, distinct recipients, mints, fan-out."""
    arr = pa.array(contracts)
    transfers: dict[str, int] = defaultdict(int)
    mints: dict[str, int] = defaultdict(int)
    recips: dict[str, set[bytes]] = defaultdict(set)
    for cd in chunk_dirs(RAW / chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        for b in pq.ParquetFile(lp).iter_batches(batch_size=1_000_000,
                                                 columns=["address", "topic1", "topic2"]):
            addr = pc.utf8_lower(b.column("address"))
            sub = b.filter(pc.is_in(addr, value_set=arr))
            if not sub.num_rows:
                continue
            for a, f, r in zip(pc.utf8_lower(sub.column("address")).to_pylist(),
                               sub.column("topic1").to_pylist(), sub.column("topic2").to_pylist()):
                transfers[a] += 1
                if f and bytes.fromhex(f[-40:]) == ZERO20:
                    mints[a] += 1
                if r:
                    rb = bytes.fromhex(r[-40:])
                    if rb != ZERO20:
                        recips[a].add(rb)
    out = {}
    for a in contracts:
        t, n = transfers.get(a, 0), len(recips.get(a, ()))
        out[a] = {"transfers": t, "distinct_recipients": n, "fanout": (t / n) if n else None,
                  "mint_share": (mints.get(a, 0) / t) if t else None}
    return out


def summarise(rows: dict[str, dict]) -> dict:
    present = [r for r in rows.values() if r["transfers"] > 0]
    phis = sorted(r["fanout"] for r in present if r["fanout"] is not None)
    return {
        "controls": len(rows),
        "present_in_corpus": len(present),
        "metadata_resolved": sum(1 for r in rows.values() if r["strength"] is not None),
        "flagged_strong": sum(1 for r in rows.values() if r["strength"] == "strong"),
        "weak": sum(1 for r in rows.values() if r["strength"] == "weak"),
        "fanout_median": statistics.median(phis) if phis else None,
        "fanout_below_2": sum(1 for p in phis if p < 2),
        "fanout_measured": len(phis),
        "ledger": dict(sorted(Counter(r["ledger"] for r in rows.values()).items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = OUT_ROOT / f"negative_controls_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")
    snap = json.loads(SNAPSHOT.read_text())
    kinds = {a: "nft" for a in nft_controls(snap, args.chain)}
    for a in registry.load_token_registry(args.chain).legit_addresses:
        kinds.setdefault("0x" + a.lower(), "registry")
    if not kinds:
        raise SystemExit("no controls for this chain")
    contracts = sorted(kinds)

    md = token_meta.fetch_metadata(contracts, rpc_url=token_meta.rpc_for(args.chain))
    t0 = time.time()
    shape = addressing(args.chain, contracts)

    block = fp.corpus_end_block(args.chain)
    eps = fp.ARCHIVE_ENDPOINTS.get(args.chain, []) + token_meta.rpc_chain_for(args.chain)
    eps = fp.archive_capable(list(dict.fromkeys(eps)), hex(block), contracts[0])
    if not eps:
        raise SystemExit(f"no archive endpoint for {args.chain} at block {block:,}")
    cal = {e: fp.calibrate_batch(e, hex(block), contracts[0]) for e in eps}
    eps = sorted((e for e in eps if cal[e] > 0), key=lambda e: -cal[e])
    ledger = fp.probe(contracts, eps, fp.probe_addresses(fp.PROBE_SEED), min(8, cal[eps[0]]), 0.6, hex(block))

    rows = {}
    for a in contracts:
        m = md.get(a) or {}
        name, symbol = m.get("name") or "", m.get("symbol") or ""
        resolved = bool(name.strip() or symbol.strip())
        v = ml.classify(name, symbol) if resolved else None
        rows[a] = {"kind": kinds[a], "name": name, "symbol": symbol,
                   "strength": v.strength if v else None,
                   "cta_path": rp.shortcut_kind(v) if v else None,
                   **shape[a], "ledger": ledger[a]["verdict"], "allowance": ledger[a]["allowance"]}
    coverage = sum(1 for r in rows.values() if r["strength"] is not None) / len(rows)
    doc = {"chain": args.chain, "block": block, "snapshot": str(SNAPSHOT.relative_to(ROOT)),
           "metadata_coverage": coverage,
           "summary": {k: summarise({a: r for a, r in rows.items() if r["kind"] == k})
                       for k in ("nft", "registry") if any(r["kind"] == k for r in rows.values())},
           "per_contract": rows, "wall_seconds": round(time.time() - t0, 1)}
    target = out if coverage >= MIN_COVERAGE else out.with_name(out.stem + "_PARTIAL.json")
    target.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    target.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/negative_controls.py", "git_commit": git_commit(),
        "block": block}, indent=1) + "\n")
    for k, s in doc["summary"].items():
        print(f"  {args.chain} {k:8s} {s}")
    print(f"  -> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
