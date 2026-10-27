"""M6: rank-stratified metadata sweep -- resolve name()/symbol() for the volume head of a chain.

Why this gates everything multi-chain
-------------------------------------
The lure classifier reads a contract's reported strings, so a chain with no metadata has no
detectable channel. Before this script, `analysis/config/token_metadata.json` held **only Base**,
which is why the channel census, the fan-out ratio and the sender-field distribution were all
single-chain results, and why the manuscript had to carry that as a threat to validity. Populating
the other four chains is the precondition for removing it.

Population: rank depth, not a fixed volume floor
------------------------------------------------
Base's existing population is the volume head excluding registry-verified tokens (ranks 3-1206 of
24,655,702 contracts, floor 488,473 transfers). We reproduce that DESIGN on the other chains rather
than that THRESHOLD, because the chains differ in size by more than two orders of magnitude: a fixed
488,473-transfer floor would return almost nothing on the smallest chain and would silently make
"prevalence" mean something different per chain. Rank depth keeps the sampling depth comparable,
which is what a rank-stratified estimator needs; the transfer floor each chain's last rank
corresponds to is reported so the difference is visible rather than hidden.

Registry-verified tokens are excluded because they are the control population elsewhere in this
paper, and including them in the detector's input would let a legitimate brand be scored as a lure.

Why this runs one chain at a time
---------------------------------
`token_meta._metadata_cache_save` is a read-modify-write of a single shared JSON file with no
locking. Two concurrent chains would interleave and one would clobber the other's key. The sweep is
network-bound, so serialising costs little against the endpoints' own latency and removes the race
entirely. It does not cost minutes: see the measured rates below.

Shared-cache discipline
-----------------------
The sweep writes through `token_meta.fetch_metadata(chain=...)`, which caches into the SHARED
`analysis/config/token_metadata.json`. That is the cache's designed purpose and the write is additive
per chain, but Base's section is a live input to another analysis in this repository, so its digest is captured
before and after and any change RAISES. The resolved metadata is also copied into this paper's own
tree, so our inputs stay pinned here even if the shared cache later moves.

Inputs
------
results/<chain>_token_volumes.csv   (READ-ONLY, the address-poisoning pipeline's cached volume table)
analysis/src/registry.py

Outputs
-------
papers/lure-broadcast/results/metadata_sweep_<chain>.json (+ .manifest.json)
papers/lure-broadcast/data/token_metadata_<chain>.json    (this paper's pinned copy)

Failure modes
-------------
- Coverage below --min-coverage ABORTS and writes nothing publishable. A 3.6%-coverage sweep once
  produced a one-token "fake-token ecosystem" elsewhere in this project; the guard exists because of
  that. An RPC error is recorded, never treated as "this contract has no name".
- Base's cached section changing raises.
- A missing volume table raises rather than sweeping an empty population.

Complexity: one RPC round trip per 25 contracts, retried with endpoint rotation. The figures below
are the rates actually recorded by the published sweeps, read from `results/metadata_sweep_*.json`;
an earlier version of this docstring quoted a pilot run and understated Optimism by three orders of
magnitude, which is the failure mode this note now exists to prevent.

Measured seconds per contract, at `requested_top` 1200: zkSync 0.0727, Arbitrum 0.0904, Polygon
zkEVM 0.1269, Optimism 40.3756. The spread is 555x and it is throttling, not work -- Optimism alone
took 48,450.7 s (13.5 h) while the other three finished in under three minutes each. Budget for the
worst endpoint, never for the mean. Base is absent because its cache predates this paper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml, registry, token_meta  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
DATA_ROOT = ROOT / "papers" / "lure-broadcast" / "data"
VOLUMES = ROOT / "results"

#: Rank depth reproducing Base's existing population (ranks 3-1206 after excluding registry tokens).
DEFAULT_TOP = 1200


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def base_digest() -> str:
    """Digest of the shared cache's Base section -- a live input to another analysis that must not move."""
    p = token_meta.METADATA_CACHE
    if not p.exists():
        return "absent"
    sec = json.loads(p.read_text()).get("base", {})
    return hashlib.sha256(json.dumps(sec, sort_keys=True).encode()).hexdigest()


def rank_endpoints(chain: str, probe_contracts: list[str], batch: int) -> list[tuple[str, float, int]]:
    """Order endpoints by whether they answer a REAL batch, then by latency.

    Why this is necessary rather than tidy: `fetch_metadata` sends its entire first sweep to
    `endpoints[0]`, so a primary that fails on batched requests costs one full backoff cycle per
    batch before any fallback is tried. Measured on Optimism, `mainnet.optimism.io` returned 0 of 50
    results after 32 s of retries while `optimism-rpc.publicnode.com` returned all 50 in 0.1 s --
    roughly 26 minutes of pure backoff across a 1,200-contract sweep. The same shape is recorded for
    Base in the project notes, where a bad primary held coverage at 3.6%.

    The canary is a real batch of the configured size, not a single call: the failure is specific to
    batch size, so a one-call probe would clear a primary that then fails on every actual request.
    Endpoints answering nothing are ranked last rather than dropped, so a run never has fewer
    endpoints than configured.
    """
    calls = []
    for j, a in enumerate(probe_contracts[:batch]):
        calls.append({"jsonrpc": "2.0", "id": f"n{j}", "method": "eth_call",
                      "params": [{"to": a, "data": token_meta.NAME_SELECTOR}, "latest"]})
        calls.append({"jsonrpc": "2.0", "id": f"s{j}", "method": "eth_call",
                      "params": [{"to": a, "data": token_meta.SYMBOL_SELECTOR}, "latest"]})
    scored: list[tuple[str, float, int]] = []
    for ep in token_meta.rpc_chain_for(chain):
        t0 = time.time()
        resp = token_meta._rpc_batch(calls, ep, retries=1)
        el = time.time() - t0
        ok = sum(1 for r in resp if isinstance(r, dict) and r.get("id") and "result" in r)
        scored.append((ep, round(el, 2), ok))
    scored.sort(key=lambda s: (0 if s[2] else 1, -s[2], s[1]))
    return scored


def load_volume_head(chain: str, top: int) -> tuple[list[str], int, int]:
    """Top `top` contracts by transfer count, registry tokens excluded.

    Returns (addresses, universe_size, floor_transfers).
    """
    # Four chains' volume tables are artefacts of the address-poisoning pipeline; Polygon zkEVM's was missing there (defect 4)
    # and was generated into THIS paper's tree, so both locations are searched. This paper's tree
    # takes precedence, so a regenerated table here can never be shadowed by a stale shared one.
    candidates = [OUT_ROOT / f"{chain}_token_volumes.csv", VOLUMES / f"{chain}_token_volumes.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise SystemExit("missing volume table, looked in: "
                         + ", ".join(str(p.relative_to(ROOT)) for p in candidates)
                         + " -- generate it first; refusing to sweep an unenumerated population")
    reg = {("0x" + a.lower()) for a in registry.load_token_registry(chain).legit_addresses}
    rows: list[tuple[str, int]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            a = r["token"].lower()
            if a in reg:
                continue
            rows.append((a, int(r["count"])))
    universe = len(rows)
    rows.sort(key=lambda kv: -kv[1])
    head = rows[:top]
    if not head:
        raise SystemExit(f"{chain}: empty volume head")
    return [a for a, _ in head], universe, head[-1][1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", required=True)
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"metadata_sweep_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    before = base_digest()
    head, universe, floor = load_volume_head(args.chain, args.top)
    ranked = rank_endpoints(args.chain, head, args.batch)
    eps = [e for e, _, _ in ranked]
    # Passing rpc_url= to fetch_metadata is NOT sufficient to change which endpoint the first sweep
    # uses. It builds its own list from rpc_chain_for(chain) and only prepends rpc_url when that URL
    # is ABSENT from the list; a ranked endpoint that is already a configured fallback is therefore
    # ignored, and the whole first sweep still goes to the hardcoded primary. Measured cost of that
    # on Optimism: 13.5 hours, against 108 s on Arbitrum whose primary happens to work.
    # The ranking is applied by scoping an override of the module's own tables for the duration of
    # the sweep, then restoring them, so no shared-library behaviour is changed for other callers.
    _saved = (token_meta.CHAIN_RPC.get(args.chain),
              token_meta.CHAIN_RPC_FALLBACKS.get(args.chain))
    token_meta.CHAIN_RPC[args.chain] = eps[0]
    token_meta.CHAIN_RPC_FALLBACKS[args.chain] = eps[1:]
    print(f"  {args.chain}: sweeping {len(head)} contracts "
          f"(volume head of {universe:,}; floor {floor:,} transfers)")
    for e, el, ok in ranked:
        print(f"    endpoint {e:44s} {ok:3d}/{2*min(args.batch, len(head))} in {el:6.2f}s")
    if ranked[0][2] == 0:
        raise SystemExit(f"no endpoint answered a {args.batch}-contract batch for {args.chain}; "
                         f"refusing to start a sweep that would spend its whole budget in backoff")

    t0 = time.time()
    try:
        md = token_meta.fetch_metadata(head, rpc_url=eps[0], batch=args.batch,
                                       sleep_s=args.sleep, chain=args.chain)
    finally:
        if _saved[0] is not None:
            token_meta.CHAIN_RPC[args.chain] = _saved[0]
        if _saved[1] is not None:
            token_meta.CHAIN_RPC_FALLBACKS[args.chain] = _saved[1]
    el = time.time() - t0
    cov = token_meta.coverage(md)

    after = base_digest()
    if before != after:
        raise SystemExit("the shared cache's Base section CHANGED during this sweep. Base is a live "
                         "input to another analysis in this repository; investigate before trusting anything.")

    # Classify, so the deliverable is the lure population and not just raw strings.
    bands: dict[str, int] = {}
    lure: dict[str, dict] = {}
    campaigns: dict[str, int] = {}
    for a, rec in md.items():
        v = ml.classify(rec.get("name"), rec.get("symbol"))
        bands[v.strength] = bands.get(v.strength, 0) + 1
        if v.strength == "strong":
            lure[a] = rec
            k = ml.campaign_key(v)
            if k:
                campaigns[k] = campaigns.get(k, 0) + 1

    doc = {
        "chain": args.chain,
        "design": ("rank depth, registry-verified tokens excluded -- reproduces Base's existing "
                   "population design rather than its absolute transfer floor, because the chains "
                   "differ in size by more than two orders of magnitude"),
        "requested_top": args.top,
        "swept": len(md),
        "universe_contracts": universe,
        "floor_transfers": floor,
        "coverage": cov,
        "min_coverage": args.min_coverage,
        "wall_seconds": round(el, 1),
        "seconds_per_contract": round(el / max(1, len(head)), 4),
        "endpoints_ranked": [{"url": e, "canary_seconds": el, "canary_answered": ok}
                             for e, el, ok in ranked],
        "bands": dict(sorted(bands.items())),
        "lure_contracts": len(lure),
        "campaigns": len(campaigns),
        "campaign_sizes": dict(sorted(campaigns.items(), key=lambda kv: -kv[1])),
    }

    # The coverage guard runs BEFORE anything publishable is written.
    if cov < args.min_coverage:
        part = OUT_ROOT / f"metadata_sweep_{args.chain}_PARTIAL.json"
        doc["ABORTED"] = (f"coverage {cov:.4f} < {args.min_coverage}; a low-coverage sweep makes the "
                          f"lure population an artefact of RPC luck rather than a measurement")
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        raise SystemExit(f"ABORT: {args.chain} coverage {cov:.4f} below --min-coverage "
                         f"{args.min_coverage}. Wrote {part.name} (unusable by make_macros). Add "
                         f"endpoints or raise the floor; do NOT lower the guard.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (DATA_ROOT / f"token_metadata_{args.chain}.json").write_text(
        json.dumps(md, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"metadata_sweep_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/run_metadata_sweep.py",
        "git_commit": git_commit(),
        "shared_cache_base_digest": before,
    }, indent=1) + "\n")

    print(f"  coverage {cov:.4f} (guard {args.min_coverage}); {el:.0f}s, "
          f"{doc['seconds_per_contract']:.4f} s/contract")
    print(f"  bands {doc['bands']}")
    print(f"  lure contracts {len(lure)} in {len(campaigns)} campaigns")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
