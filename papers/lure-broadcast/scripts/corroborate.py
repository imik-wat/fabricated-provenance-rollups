"""Corroborate the lure detector against independent public label sources, each with a control.

Why this exists
---------------
The lure label would otherwise be circular: a lure is whatever the detector says it is. This script
tests the detector's output against labels produced by other people, and -- non-negotiably -- against a
matched control drawn from the same population, because a raw agreement rate has no meaning without a
base rate. The project has already caught itself once on exactly this: a raw 16.05% recipient overlap
looked publishable until a control drawn from the identical parquet row groups showed the base rate was
10.82%, a ratio of only 1.48x.

Two of the three K3 kill-condition legs are evaluated here (the third, two-rater kappa, needs humans):
  contract-level  -- flagged contracts vs Covalent GoldRush / Alchemy spam lists
  domain-level    -- extracted campaign domains vs the MetaMask eth-phishing-detect blocklist

Forta is deliberately absent: no API key is obtainable. See papers/lure-broadcast/docs/DEFECTS.md and
the ground-truth section of the plan.

The control, and why it is rank-matched
--------------------------------------
Lure prevalence is strongly rank-dependent: measured 36.6% in Optimism's top 500 and 0/50 in Base's
25,000-94,458 band. A control sampled uniformly from the whole token universe would therefore differ
from the treatment group in volume rank, and volume rank predicts spam-list membership independently of
lureness -- the ratio would be confounded rather than wrong-looking. So the control is drawn from the
SAME volume-rank band as the treatment contracts and excludes them.

Inputs
------
analysis/config/token_metadata.json                      (cached name/symbol; base only at present)
results/<chain>_token_volumes.csv                        (rank frame, from the address-poisoning pipeline; read-only)
papers/lure-broadcast/data/labels/**                      (pinned label snapshots)

Outputs
-------
papers/lure-broadcast/results/corroboration_<chain>.json
papers/lure-broadcast/results/corroboration_<chain>.manifest.json

Failure modes
-------------
- A missing label snapshot RAISES; it is never treated as "no labels matched", which would read as a
  detector failure rather than a missing input.
- A control pool too small for the requested size raises inside controls.matched_control rather than
  silently returning a short control, which would inflate the ratio.
- p_value underflows to 0.0 at large n; it is emitted as a float but the manifest records
  `p_value_underflow: true` so a table generator renders "p < 1e-300" rather than "p = 0".
- ratio can be inf when the control has zero hits; recorded explicitly rather than formatted away.

Complexity: O(labels + contracts); the Base Covalent list is 1.4M addresses (~74 MB) and loads in a few
seconds. Memory a few hundred MB.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import controls, metadata_lure as ml  # noqa: E402

LABELS = ROOT / "papers" / "lure-broadcast" / "data" / "labels"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
METADATA_CACHE = ROOT / "analysis" / "config" / "token_metadata.json"

# Which pinned source covers which chain, per the M0 gate. Polygon zkEVM has none.
COVALENT_ERC20 = {
    "base": ["base_mainnet_token_spam_contracts_yes.yaml", "base_mainnet_token_spam_contracts_maybe.yaml"],
    "optimism": ["op_mainnet_token_spam_contracts_yes.yaml", "op_mainnet_token_spam_contracts_maybe.yaml"],
}
ALCHEMY_CHAINS = {"arbitrum-one", "zksync-era"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_covalent(filename: str) -> set[str]:
    """Addresses from a GoldRush YAML list.

    Entries are `chainId/address/typeCode` (e.g. `8453/0x0035.../22`), so they are split rather than
    regex-matched as bare addresses. The Base `yes` list carries 1,403,708 items but only 1,400,648
    unique addresses -- 3,060 duplicates -- so a set is the right container and uniqueness must not be
    assumed.
    """
    path = LABELS / "covalent" / filename
    if not path.exists():
        raise SystemExit(f"missing pinned label file {path.relative_to(ROOT)}; run fetch_labels.py")
    out: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("- "):
                continue
            parts = line[2:].split("/")
            if len(parts) >= 2 and parts[1].startswith("0x"):
                out.add(parts[1].lower())
    if not out:
        raise SystemExit(f"{filename} parsed to zero addresses -- the format has changed; refusing "
                         f"to report a zero corroboration rate as a finding")
    return out


def load_alchemy(chain: str) -> set[str]:
    path = LABELS / "alchemy" / f"{chain}.json"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}; run fetch_labels.py")
    data = json.loads(path.read_text())
    addrs = data.get("contractAddresses") if isinstance(data, dict) else data
    if not addrs:
        raise SystemExit(f"{path.name} contains no addresses")
    return {a.lower() for a in addrs}


def load_metamask_domains() -> set[str]:
    """Blocklisted domains. NOTE the legacy key is `blacklist`, not `blocklist`."""
    path = LABELS / "metamask" / "config.json"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}; run fetch_labels.py")
    cfg = json.loads(path.read_text())
    if "blacklist" not in cfg:
        raise SystemExit("metamask config has no 'blacklist' key -- schema changed; do not silently "
                         "read an absent key as an empty blocklist")
    return {d.lower().strip() for d in cfg["blacklist"] if d}


def rank_frame(chain: str) -> list[str]:
    """Contract addresses ordered by descending transfer count (the rank frame).

    Prefers the new tree's census, falls back to the address-poisoning pipeline's artefact (read-only) for chains this paper
    has not yet re-censused.
    """
    for cand in (OUT_ROOT / f"{chain}_token_volumes.csv", ROOT / "results" / f"{chain}_token_volumes.csv"):
        if cand.exists():
            with cand.open() as fh:
                return [row["token"].lower() for row in csv.DictReader(fh)]
    raise SystemExit(f"no token_volumes.csv for {chain} in either tree")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chain", choices=["base", "optimism", "arbitrum-one", "zksync-era", "polygon-zkevm"])
    ap.add_argument("--control-multiple", type=float, default=5.0,
                    help="control size as a multiple of the treatment size (default 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.chain == "polygon-zkevm":
        raise SystemExit("polygon-zkevm has NO external label source (see the M0 gate). Its prevalence "
                         "figures rest on annotator agreement and internal controls only; that is a "
                         "limitation to disclose, not something this script can supply.")

    out_path = OUT_ROOT / f"corroboration_{args.chain}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path.relative_to(ROOT)} exists; pass --force")

    md_all = json.loads(METADATA_CACHE.read_text())
    md = md_all.get(args.chain)
    if not md:
        raise SystemExit(f"no cached metadata for {args.chain} (have: {sorted(md_all)}). The rank-"
                         f"stratified metadata sweep (M6) must run first for this chain.")

    # Classify every contract we have metadata for.
    flagged: list[str] = []
    unflagged: list[str] = []
    domains: set[str] = set()
    for addr, m in md.items():
        v = ml.classify(m.get("name"), m.get("symbol"))
        (flagged if v.strength == "strong" else unflagged).append(addr.lower())
        if v.strength == "strong":
            domains.update(v.domains)
    if not flagged:
        raise SystemExit(f"{args.chain}: no contracts flagged; nothing to corroborate")

    # Rank-matched control: same volume-rank band as the treatment, treatment excluded.
    frame = rank_frame(args.chain)
    rank = {a: i for i, a in enumerate(frame)}
    ranks = [rank[a] for a in flagged if a in rank]
    if not ranks:
        raise SystemExit(f"{args.chain}: flagged contracts are absent from the rank frame -- the "
                         f"metadata cache and token_volumes.csv disagree; refusing to build a control")
    lo, hi = min(ranks), max(ranks)
    pool = [a for a in frame[lo:hi + 1] if a not in set(flagged)]
    size = min(int(len(flagged) * args.control_multiple), len(pool))
    control = controls.matched_control(pool, exclude=set(flagged), size=size, seed=args.seed)

    result: dict = {
        "chain": args.chain,
        "metadata_contracts": len(md),
        "flagged_strong": len(flagged),
        "rank_band": [lo, hi],
        "control_size": len(control),
        "control_multiple_requested": args.control_multiple,
        "distinct_campaign_domains": len(domains),
        "axes": {},
        "note": ("Every ratio is treatment-rate / control-rate against a rank-matched control. A ratio "
                 "quoted without both rates is not a measurement."),
    }

    # ---- axis 1: contract-level spam lists -------------------------------------------------
    references: dict[str, set[str]] = {}
    if args.chain in COVALENT_ERC20:
        for fn in COVALENT_ERC20[args.chain]:
            tier = "yes" if "_yes" in fn else "maybe"
            references[f"covalent_erc20_{tier}"] = load_covalent(fn)
    if args.chain in ALCHEMY_CHAINS:
        references["alchemy_spam"] = load_alchemy(args.chain)

    for label, ref in references.items():
        r = controls.overlap_ratio(flagged, control, ref, seed=args.seed)
        result["axes"][label] = {
            "reference_size": len(ref),
            "n_treatment": r.n_treatment, "hits_treatment": r.hits_treatment,
            "rate_treatment": r.rate_treatment,
            "n_control": r.n_control, "hits_control": r.hits_control,
            "rate_control": r.rate_control,
            "ratio": (None if math.isinf(r.ratio) else r.ratio),
            "ratio_is_infinite": math.isinf(r.ratio),
            "ci95": [r.ci95_low, r.ci95_high],
            "p_value": r.p_value,
            "p_value_underflow": (r.p_value == 0.0),
        }
        print(f"  {label:24s} treat {r.rate_treatment:.4f} ({r.hits_treatment}/{r.n_treatment})  "
              f"ctrl {r.rate_control:.4f} ({r.hits_control}/{r.n_control})  ratio "
              f"{r.ratio:.2f}  CI[{r.ci95_low:.2f},{r.ci95_high:.2f}]")

    # ---- axis 2: domain blocklist ----------------------------------------------------------
    # The control here is domains extracted from NON-flagged contracts (i.e. legitimate branding such
    # as curve.fi). It is expected to be small; if it is too small to resolve a base rate that is
    # reported, not hidden, because an unresolvable base rate means the ratio is uninterpretable.
    blocked = load_metamask_domains()
    ctrl_domains: set[str] = set()
    for addr in unflagged:
        m = md.get(addr) or md.get(addr.lower()) or {}
        v = ml.classify(m.get("name"), m.get("symbol"))
        ctrl_domains.update(v.domains)
        ctrl_domains.update(v.benign_domains)
    t_hits = sum(1 for d in domains if d in blocked)
    c_hits = sum(1 for d in ctrl_domains if d in blocked)
    result["axes"]["metamask_domain_blocklist"] = {
        "reference_size": len(blocked),
        "n_treatment": len(domains), "hits_treatment": t_hits,
        "rate_treatment": (t_hits / len(domains)) if domains else None,
        "n_control": len(ctrl_domains), "hits_control": c_hits,
        "rate_control": (c_hits / len(ctrl_domains)) if ctrl_domains else None,
        "control_resolvable": len(ctrl_domains) >= 30,
        "caveat": ("Control = domains in the metadata of non-flagged contracts. If control_resolvable "
                   "is false the base rate is unresolved and the ratio must not be quoted."),
    }
    print(f"  {'metamask_domains':24s} treat {t_hits}/{len(domains)}  ctrl {c_hits}/{len(ctrl_domains)}"
          f"  (control resolvable: {len(ctrl_domains) >= 30})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest = {
        "output": out_path.name,
        "producer": "papers/lure-broadcast/scripts/corroborate.py",
        "git_commit": git_commit(),
        "label_manifest": str((LABELS / "manifest.json").relative_to(ROOT)),
        "seed": args.seed,
        "forta": "unavailable -- no API key obtainable; see docs/DEFECTS.md",
    }
    (OUT_ROOT / f"corroboration_{args.chain}.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    print(f"\n  -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
