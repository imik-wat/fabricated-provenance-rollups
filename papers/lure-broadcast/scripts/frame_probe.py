"""The ledger and authorisation probes over EVERY examined contract, on every chain (IEEE R1.Q2, R2.1; WP3).

The published probe (forgery_probe.py) ran on the 141 flagged Base contracts, with six registry tokens
as controls. Reviewer 1 asked for stronger evidence behind the fabricated-provenance classification,
and Reviewer 2 for validation that does not depend on the codebook. Probing the whole frame answers
both:
  * every unflagged frame contract becomes a control, about a thousand per chain instead of six;
  * a detector-flag x ledger-verdict table compares the rule against contract STATE, which shares
    nothing with the codebook, the raters or the lexicon;
  * the other four chains get the contract-state evidence that only Base had.

The method is the published one, imported unchanged: the same two seeded probe addresses (verified
absent from every chain by forgery_chain_scan.py), the same verdict rule, the same block pinning to
the corpus end, the same archive canary and batch calibration. Nothing is re-implemented. The only
addition is free archive endpoints for chains whose configured ones do not serve historical state,
kept here so the shared configuration is untouched.

Guards (raise or mark partial, never report quietly):
  * a registry control returning a fabricated verdict raises, because then the probe would not
    discriminate;
  * no archive-capable endpoint raises, because falling back to "latest" would make the run
    irreproducible;
  * more than MAX_NO_RESPONSE of the frame unanswered writes frame_probe_<chain>_PARTIAL.json,
    which nothing downstream reads;
  * on Base, any flagged contract whose verdict differs from the published forgery_probe_base.json
    at the same block raises.

Output: papers/lure-broadcast/results/frame_probe_<chain>.json (+ .manifest.json)
Complexity: 4 eth_calls per contract, batched; ~1,200 contracts per chain, minutes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, registry, token_meta  # noqa: E402
import forgery_probe as fp  # noqa: E402
from build_annotation_sample_v2 import is_unresolved, metadata_path  # noqa: E402

OUT_ROOT = PAPER / "results"
#: Free endpoints measured on 2026-10-27 to serve archive state at the corpus end block, for chains
#: whose configured endpoints do not. Arbitrum: arb1.arbitrum.io, drpc, publicnode, llamarpc, 1rpc
#: and meowrpc all refused; these two answered.
EXTRA_ARCHIVE = {
    "arbitrum-one": ["https://arbitrum-one.public.blastapi.io", "https://arb-pokt.nodies.app"],
}
MAX_NO_RESPONSE = 0.05
FABRICATED = ("constant_nonzero", "address_derived")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def frame_bands(md: dict) -> dict[str, str]:
    """Detector band for every resolved frame contract (lower-case address -> strong/weak/none)."""
    return {a.lower(): ml.classify(e.get("name"), e.get("symbol")).strength
            for a, e in md.items() if isinstance(e, dict) and not is_unresolved(e)}


def cross_table(bands: dict[str, str], verdicts: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Detector band x ledger verdict, counting every contract exactly once."""
    table: dict[str, Counter] = {}
    for a, band in bands.items():
        table.setdefault(band, Counter())[verdicts[a]["verdict"]] += 1
    return {b: dict(sorted(c.items())) for b, c in sorted(table.items())}


def summary(table: dict[str, dict[str, int]]) -> dict:
    """Flagged (strong) against not flagged, on contracts that answered."""
    def side(bands):
        fab = sum(table.get(b, {}).get(v, 0) for b in bands for v in FABRICATED)
        led = sum(table.get(b, {}).get("zero_ledger", 0) for b in bands)
        return {"fabricated": fab, "zero_ledger": led, "answered": fab + led}
    return {"flagged": side(["strong"]), "not_flagged": side(["weak", "none"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = OUT_ROOT / f"frame_probe_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    raw = json.loads(metadata_path(args.chain).read_text())
    md = raw[args.chain] if args.chain == "base" else raw
    bands = frame_bands(md)
    reg = sorted("0x" + a.lower() for a in registry.load_token_registry(args.chain).legit_addresses)
    controls = [a for a in reg if a not in bands]
    if not controls:
        raise SystemExit("no registry controls outside the frame; the probe would be uncontrolled")

    probes = fp.probe_addresses(fp.PROBE_SEED)
    block = fp.corpus_end_block(args.chain)
    block_hex = hex(block)
    eps_all = (EXTRA_ARCHIVE.get(args.chain, []) + fp.ARCHIVE_ENDPOINTS.get(args.chain, [])
               + token_meta.rpc_chain_for(args.chain))
    eps = fp.archive_capable(list(dict.fromkeys(eps_all)), block_hex, controls[0])
    if not eps:
        raise SystemExit(f"no endpoint serves archive state for {args.chain} at block {block:,}")
    cal = {e: fp.calibrate_batch(e, block_hex, controls[0]) for e in eps}
    eps = sorted((e for e in eps if cal[e] > 0), key=lambda e: -cal[e])
    if not eps:
        raise SystemExit("no archive endpoint answers a single-contract batch")
    batch = min(args.batch, cal[eps[0]])
    print(f"  {args.chain}: {len(bands)} frame contracts, {len(controls)} controls, block {block:,}, "
          f"endpoints {cal}, batch {batch}")

    t0 = time.time()
    res = fp.probe(sorted(bands), eps, probes, batch, args.sleep, block_hex)
    ctrl = fp.probe(controls, eps, probes, batch, args.sleep, block_hex)
    wall = time.time() - t0

    bad = [a for a, v in ctrl.items() if v["verdict"] in FABRICATED]
    if bad:
        raise SystemExit(f"CONTROL FAILURE on {args.chain}: {bad[:3]}. The probe does not discriminate.")
    if args.chain == "base":
        published = json.loads((OUT_ROOT / "forgery_probe_base.json").read_text())["per_contract"]
        diff = [a for a, v in published.items()
                if a in res and res[a]["verdict"] != v["verdict"]
                and "no_response" not in (res[a]["verdict"], v["verdict"])]
        if diff:
            raise SystemExit(f"verdicts differ from the published Base probe at the same block: {diff[:5]}")

    table = cross_table(bands, res)
    no_resp = sum(1 for v in res.values() if v["verdict"] == "no_response")
    partial = no_resp / len(res) > MAX_NO_RESPONSE
    doc = {
        "chain": args.chain, "block": block, "probe_seed": fp.PROBE_SEED, "probe_addresses": probes,
        "frame_contracts": len(bands), "no_response": no_resp, "partial": partial,
        "band_by_verdict": table, "flagged_vs_not": summary(table),
        "controls": {"n": len(controls), "verdicts": dict(Counter(v["verdict"] for v in ctrl.values()))},
        "allowance_by_band": {b: dict(Counter(res[a]["allowance"] for a in bands if bands[a] == b))
                              for b in sorted(set(bands.values()))},
        "fabricated_not_flagged": sorted(a for a in bands
                                         if bands[a] != "strong" and res[a]["verdict"] in FABRICATED),
        "per_contract": {a: {"band": bands[a], **{k: res[a][k] for k in ("verdict", "allowance",
                                                                          "balance_equals_total_supply")}}
                         for a in sorted(bands)},
        "endpoints": cal, "wall_seconds": round(wall, 1),
    }
    target = out.with_name(out.stem + "_PARTIAL.json") if partial else out
    target.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    target.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/frame_probe.py", "git_commit": git_commit(),
        "block": block}, indent=1) + "\n")
    s = doc["flagged_vs_not"]
    print(f"  flagged: {s['flagged']}  not flagged: {s['not_flagged']}  no_response {no_resp}"
          f"{'  PARTIAL' if partial else ''}  ({wall:.0f}s) -> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
