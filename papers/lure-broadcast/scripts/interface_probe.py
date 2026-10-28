"""Is an empty data field enough to call a transfer non-fungible? (reviewer round 2, item 13.)

The question
------------
The census separates the token standards by the width of the event's data field. That is a
syntactic test standing in for a semantic claim, and the reviewer is right to ask what it costs. The
honest answer cannot be argued; it has to be measured against the interface the contracts actually
declare.

This script measures it. It reads the empty-data volume of every contract on a chain in one full
pass, takes the contracts carrying the most of it, and asks each one directly:

  supportsInterface(0x80ac58cd)  -- does it claim ERC-721?
  supportsInterface(0xd9b67a26)  -- does it claim ERC-1155?
  decimals()                     -- does it behave like a fungible token?

A contract answering `decimals()` while claiming no non-fungible interface is evidence that the
empty-data class is not the non-fungible class. Results are reported BOTH by contract and weighted
by the empty-data rows each contract emitted, because the two differ: a handful of high-volume
contracts can dominate the weighted figure while being a small share of the contracts.

What this can and cannot establish
----------------------------------
It bounds the discriminant's error over the volume head, which is where the volume is. It says
nothing about the tail, and it cannot be turned into a corrected census: reclassifying rows would
require probing every contract, and a probe answers for the implementation reachable now rather than
the one that emitted the event. A proxy may have been upgraded in between. The figure is therefore
reported as a measured caveat on a syntactic class, never as a correction factor.

An absent or throttled answer is recorded as `unresolved` and counted in its own bucket. It is never
folded into either of the informative classes, since the direction of that error is unknown.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet  (READ-ONLY)

Outputs
-------
papers/lure-broadcast/results/interface_probe_<chain>.json (+ .manifest.json)

Failure modes
-------------
- If every probed contract is unresolved the run RAISES: an all-unresolved probe measures the
  endpoint, not the population, and publishing it as a coverage figure would be a fabrication.
- The empty-data row total is asserted against the census; a divergence RAISES.

Complexity: one full pass over two columns, parallel by row group; then 3 eth_calls per contract.
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

from src import token_meta  # noqa: E402
from rpc_answers import TRANSPORT, element_kind  # noqa: E402
from src.nonfungible import NONFUNGIBLE_DATA_HEXLEN  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
MAX_WORKERS = 12
ZK_MAX_WORKERS = 4

#: ERC-165 `supportsInterface(bytes4)` with the two interface identifiers, and `decimals()`.
SEL_SUPPORTS = "0x01ffc9a7"
IFACE_ERC721 = "80ac58cd"
IFACE_ERC1155 = "d9b67a26"
SEL_DECIMALS = "0x313ce567"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _scan(path_rg):
    """Empty-data row count per emitting contract, for one row group."""
    path, rg = path_rg
    out: Counter = Counter()
    total = 0
    tbl = pq.ParquetFile(path).read_row_group(rg, columns=["address", "data"])
    for b in tbl.to_batches(max_chunksize=1_000_000):
        m = pc.equal(pc.utf8_length(b.column("data")), NONFUNGIBLE_DATA_HEXLEN)
        if not pc.any(m).as_py():
            continue
        sub = pc.utf8_lower(b.column("address").filter(m))
        total += len(sub)
        for st in pc.value_counts(sub):
            out[st["values"].as_py()] += st["counts"].as_py()
    return out, total


def _bool_word(hexstr: str | None) -> bool | None:
    """Decode an ABI bool return. None if the call gave nothing decodable."""
    if not hexstr or not isinstance(hexstr, str):
        return None
    h = hexstr.removeprefix("0x")
    if len(h) < 64:
        return None
    return int(h[:64], 16) == 1


def classify_interface(is721: bool | None, is1155: bool | None,
                       decimals_ok: bool) -> str:
    """Verdict for one contract. Pure, so the classification is testable without a network.

    `unresolved` is returned whenever the ERC-165 answers are absent, because a contract that did
    not answer is not evidence either way and must not be counted as a confirmation or a refutation.
    A denial needs BOTH interfaces answered false: a False beside a None is not a denial of both,
    and reading it as one is how a throttled supportsInterface beside an answered one manufactured
    denials in the 2026-10-28 re-run.
    """
    if is721:
        return "erc721_confirmed"
    if is1155:
        return "erc1155_confirmed"
    if is721 is False and is1155 is False:
        return "answers_decimals_not_nonfungible" if decimals_ok else "denies_nonfungible_interface"
    # no complete ERC-165 answer
    return "answers_decimals_no_interface" if decimals_ok else "unresolved"


def probe(contracts: list[str], endpoints: list[str], batch: int = 20) -> dict[str, dict]:
    """ERC-165 and decimals() answers per contract.

    A contract is classified only from three answers the CONTRACT gave (rpc_answers.element_kind:
    a result or an execution error). One with any transport failure is asked again on the next
    endpoint, and one that no endpoint answers in full is `unresolved` with no answers recorded.
    Accepting the first endpoint that returned any result, as this did until the 2026-10-28 re-run,
    kept partly throttled batches whole.
    """
    out: dict[str, dict] = {}
    for i in range(0, len(contracts), batch):
        pending = list(contracts[i:i + batch])
        for ep in endpoints:
            if not pending:
                break
            pending = _probe_on(pending, ep, out)
        for a in pending:
            out[a] = {"verdict": "unresolved", "erc721": None, "erc1155": None, "decimals": False}
        print(f"    probed {min(i + batch, len(contracts))}/{len(contracts)}", flush=True)
    return out


def _probe_on(chunk: list[str], ep: str, out: dict[str, dict]) -> list[str]:
    """Ask one endpoint about `chunk`; record the contracts answered in full into `out` and return
    the rest."""
    calls = []
    for j, a in enumerate(chunk):
        addr = "0x" + a.removeprefix("0x")
        calls.append({"jsonrpc": "2.0", "id": f"a{j}", "method": "eth_call",
                      "params": [{"to": addr,
                                  "data": SEL_SUPPORTS + IFACE_ERC721 + "0" * 56}, "latest"]})
        calls.append({"jsonrpc": "2.0", "id": f"b{j}", "method": "eth_call",
                      "params": [{"to": addr,
                                  "data": SEL_SUPPORTS + IFACE_ERC1155 + "0" * 56}, "latest"]})
        calls.append({"jsonrpc": "2.0", "id": f"c{j}", "method": "eth_call",
                      "params": [{"to": addr, "data": SEL_DECIMALS}, "latest"]})
    by_id = {x.get("id"): x for x in token_meta._rpc_batch(calls, ep) if isinstance(x, dict)}
    still = []
    for j, a in enumerate(chunk):
        els = [by_id.get(f"{k}{j}") for k in "abc"]
        if any(element_kind(e) == TRANSPORT for e in els):
            still.append(a)
            continue
        is721, is1155 = (_bool_word(e.get("result")) for e in els[:2])
        dec = els[2].get("result")
        dec_ok = bool(dec) and dec != "0x" and len(dec.removeprefix("0x")) >= 64
        out[a] = {"verdict": classify_interface(is721, is1155, dec_ok),
                  "erc721": is721, "erc1155": is1155, "decimals": dec_ok}
    return still


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"interface_probe_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cap = ZK_MAX_WORKERS if args.chain == "zksync-era" else MAX_WORKERS
    workers = args.workers or max(1, min(cap, (os.cpu_count() or 4) - 2))
    tasks = []
    for cd in sorted((RAW / args.chain).glob("chunk_*")):
        lp = cd / "logs.parquet"
        if lp.exists() and (cd / "_DONE").exists():
            tasks += [(str(lp), i) for i in range(pq.ParquetFile(lp).num_row_groups)]

    per: Counter = Counter()
    total = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c, t in ex.map(_scan, tasks, chunksize=4):
            per.update(c)
            total += t
    print(f"  {total:,} empty-data rows from {len(per):,} contracts in {(time.time()-t0)/60:.1f} min")

    census = json.loads((OUT_ROOT / f"{args.chain}_nf_census.json").read_text())
    if total != census["nonfungible_rows"]:
        raise SystemExit(f"empty-data rows found ({total}) disagree with the census "
                         f"({census['nonfungible_rows']}); the same discriminant over the same "
                         f"corpus must give the same answer")

    top = [a for a, _ in per.most_common(args.top)]
    eps = token_meta.rpc_chain_for(args.chain)
    verdicts = probe(top, eps)

    tally, weighted = Counter(), Counter()
    for a, v in verdicts.items():
        tally[v["verdict"]] += 1
        weighted[v["verdict"]] += per[a]
    probed_rows = sum(per[a] for a in top)
    if tally.get("unresolved", 0) == len(top):
        raise SystemExit("every probed contract was unresolved; this measures the endpoint rather "
                         "than the population and must not be reported as coverage")

    not_nf = ("answers_decimals_no_interface", "answers_decimals_not_nonfungible")
    doc = {
        "chain": args.chain,
        "empty_data_rows": total,
        "empty_data_contracts": len(per),
        "probed_contracts": len(top),
        "probed_rows": probed_rows,
        "probed_share_of_empty_data": probed_rows / total if total else None,
        "by_verdict_contracts": dict(tally),
        "by_verdict_rows": dict(weighted),
        "answers_decimals_share_of_probed_rows":
            sum(weighted[k] for k in not_nf) / probed_rows if probed_rows else None,
        "erc721_share_of_probed_rows":
            weighted.get("erc721_confirmed", 0) / probed_rows if probed_rows else None,
        "verdicts": verdicts,
        "interpretation": ("a contract answering decimals() while claiming no non-fungible "
                           "interface shows the empty-data class is syntactic. The figure bounds "
                           "the discriminant over the volume head only, and is NOT a correction "
                           "factor: a probe answers for the implementation reachable now, which a "
                           "proxy may have changed since the event was emitted."),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"interface_probe_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/interface_probe.py",
        "git_commit": git_commit(), "top": args.top,
    }, indent=1) + "\n")
    print(f"  contracts {dict(tally)}")
    print(f"  probed rows cover {probed_rows/total*100:.2f}% of empty-data volume; "
          f"answers decimals(): {doc['answers_decimals_share_of_probed_rows']*100:.2f}% of them")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
