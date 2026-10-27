"""Snapshot and PIN the external label sources used as this paper's independent ground truth.

Why this exists
---------------
The paper's lure label must not be circular ("a lure is whatever our detector says it is"), so it is
corroborated against label sources produced by other people. Those sources are live and drift: the
Alchemy Ethereum spam list moved from 165,033 to 165,036 entries between two probes hours apart, and
MetaMask's blocklist is committed to many times a day. A paper whose numbers depend on "whatever the
endpoint said that afternoon" is not reproducible, so every source is snapshotted to disk with the
commit SHA or a content digest, the entry count, and the access date. Analysis then reads the
snapshot, never the network.

The original plan named Forta's GraphQL label feed as the primary oracle. **A Forta API key is not
obtainable, so that source is unavailable.** This does not trigger the paper's K3 kill condition,
which requires the external feed to be unreachable AND annotator agreement to fail AND blocklist
corroboration to be no better than its control. The substitutes below give WIDER chain coverage than
Forta ever did (Forta supported only Arbitrum and Optimism of the five chains studied):

  Alchemy getSpamContracts   Arbitrum, zkSync Era      (public "demo" key, no auth)
  Covalent GoldRush lists    Base, Optimism            (public repo; ERC-20 AND NFT lists)
  MetaMask eth-phishing-detect  chain-independent      (domain blocklist wallets enforce)
  Polygon zkEVM              NO external label source  -> disclose as a limitation

The MetaMask list matters most for this paper's unit of evidence: the paper clusters campaigns by the
registrable domain extracted from token metadata, so a domain-level oracle tests the same object the
detector emits, independently of any token data. It is also the list wallets actually enforce, which
makes "was this lure domain known-bad, and when" answerable rather than rhetorical.

Caveats recorded rather than smoothed over
------------------------------------------
- Alchemy zkSync Era returns EXACTLY 62,500 entries. A round number that size is far more likely a
  server-side cap than a true population, so it is flagged `suspect_capped` and must not be treated
  as a complete list. Arbitrum's 43,464 is not round and is treated as complete.
- Alchemy Optimism and Base return HTTP 500 persistently (confirmed across separate sessions), so
  those two chains come from Covalent instead. A 500 is recorded as an error, never as an empty list.
- Covalent splits ERC-20 into `_yes` and `_maybe` confidence tiers. They are kept SEPARATE; merging
  them would silently import someone else's uncertainty into our corroboration rate.

Inputs
------
Network only. No repository data is read.

Outputs
-------
papers/lure-broadcast/data/labels/<source>/...            (raw snapshots)
papers/lure-broadcast/data/labels/manifest.json           (pins, counts, access dates, errors)

Failure modes
-------------
- Any source that fails is recorded in the manifest with its HTTP status and is NOT written as an
  empty file, so a fetch failure can never be mistaken for "this chain has no spam".
- Refuses to overwrite an existing snapshot without --force, so a re-run cannot silently re-pin the
  ground truth underneath results that were computed against the previous pin.
- Exits non-zero if any source failed, so a partial snapshot is loud.

Complexity: O(downloaded bytes); a few MB per source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "papers" / "lure-broadcast" / "data" / "labels"

UA = "rollups-research/1.0 (academic measurement; contact via repository)"

# Alchemy NFT-API spam contract lists. The public "demo" key is documented for evaluation and needs
# no account; recorded here so the paper can state exactly what was queried.
ALCHEMY_NETS = {
    "arbitrum-one": "arb-mainnet",
    "zksync-era": "zksync-mainnet",
    # Queried and recorded as errors rather than omitted, so the gap is visible in the manifest:
    "optimism": "opt-mainnet",
    "base": "base-mainnet",
    "polygon-zkevm": "polygonzkevm-mainnet",
}

COVALENT_REPO = "covalenthq/goldrush-enhanced-spam-lists"
COVALENT_FILES = [
    "src/lists/erc20/base_mainnet_token_spam_contracts_yes.yaml",
    "src/lists/erc20/base_mainnet_token_spam_contracts_maybe.yaml",
    "src/lists/erc20/op_mainnet_token_spam_contracts_yes.yaml",
    "src/lists/erc20/op_mainnet_token_spam_contracts_maybe.yaml",
    "src/lists/nft/base_mainnet_nft_spam_contracts.yaml",
    "src/lists/nft/op_mainnet_nft_spam_contracts.yaml",
]

METAMASK_REPO = "MetaMask/eth-phishing-detect"
METAMASK_FILE = "src/config.json"


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def head_commit(repo: str) -> dict:
    """Current HEAD commit of a GitHub repo's default branch -- this is the pin."""
    d = json.loads(get(f"https://api.github.com/repos/{repo}/commits/HEAD"))
    return {"sha": d["sha"], "date": d["commit"]["author"]["date"]}


def write(path: Path, blob: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-pin over an existing snapshot")
    args = ap.parse_args()

    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path.relative_to(ROOT)} exists; pass --force to re-pin "
                         f"(re-pinning invalidates any result computed against the previous pin)")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    man: dict = {"accessed": now, "sources": {}, "errors": [],
                 "note": "Forta GraphQL omitted: no API key obtainable. See module docstring."}

    # ---- Alchemy -----------------------------------------------------------------------------
    alch: dict = {"endpoint": "https://<net>.g.alchemy.com/nft/v3/demo/getSpamContracts",
                  "auth": "public demo key", "chains": {}}
    for chain, net in ALCHEMY_NETS.items():
        url = f"https://{net}.g.alchemy.com/nft/v3/demo/getSpamContracts"
        try:
            blob = get(url)
            data = json.loads(blob)
            addrs = data.get("contractAddresses") if isinstance(data, dict) else data
            if not isinstance(addrs, list):
                raise ValueError(f"unexpected payload shape: {str(data)[:120]}")
            rec = write(OUT / "alchemy" / f"{chain}.json", blob)
            rec["entries"] = len(addrs)
            rec["network"] = net
            # A round 62,500 is a cap signature, not a population.
            rec["suspect_capped"] = len(addrs) % 12500 == 0
            alch["chains"][chain] = rec
            flag = "  <-- SUSPECT CAP" if rec["suspect_capped"] else ""
            print(f"  alchemy   {chain:14s} {len(addrs):>7,} entries{flag}")
        except Exception as exc:
            msg = f"alchemy {chain} ({net}): {type(exc).__name__}: {exc}"
            alch["chains"][chain] = {"error": str(exc)}
            man["errors"].append(msg)
            print(f"  alchemy   {chain:14s} FAILED: {exc}")
    man["sources"]["alchemy"] = alch

    # ---- Covalent ----------------------------------------------------------------------------
    cov: dict = {"repo": COVALENT_REPO, "files": {}}
    try:
        cov["pin"] = head_commit(COVALENT_REPO)
        for rel in COVALENT_FILES:
            url = f"https://raw.githubusercontent.com/{COVALENT_REPO}/{cov['pin']['sha']}/{rel}"
            blob = get(url)
            rec = write(OUT / "covalent" / Path(rel).name, blob)
            # These YAMLs are flat address lists; count list items without a YAML dependency.
            rec["lines"] = sum(1 for ln in blob.decode("utf-8", "replace").splitlines()
                               if ln.strip().startswith("- "))
            cov["files"][rel] = rec
            print(f"  covalent  {Path(rel).name:52s} {rec['lines']:>7,} entries")
    except Exception as exc:
        man["errors"].append(f"covalent: {type(exc).__name__}: {exc}")
        print(f"  covalent  FAILED: {exc}")
    man["sources"]["covalent"] = cov

    # ---- MetaMask ----------------------------------------------------------------------------
    mm: dict = {"repo": METAMASK_REPO, "file": METAMASK_FILE}
    try:
        mm["pin"] = head_commit(METAMASK_REPO)
        blob = get(f"https://raw.githubusercontent.com/{METAMASK_REPO}/{mm['pin']['sha']}/{METAMASK_FILE}")
        rec = write(OUT / "metamask" / "config.json", blob)
        cfg = json.loads(blob)
        # Legacy key names: blacklist/whitelist, NOT blocklist/allowlist. Recorded so downstream code
        # cannot silently read a missing key as an empty list.
        rec["keys"] = sorted(cfg)
        rec["blacklist"] = len(cfg.get("blacklist", []))
        rec["whitelist"] = len(cfg.get("whitelist", []))
        rec["fuzzylist"] = len(cfg.get("fuzzylist", []))
        rec["tolerance"] = cfg.get("tolerance")
        rec["config_version"] = cfg.get("version")
        mm["snapshot"] = rec
        print(f"  metamask  blacklist {rec['blacklist']:,} domains  "
              f"(whitelist {rec['whitelist']}, fuzzylist {rec['fuzzylist']})")
    except Exception as exc:
        man["errors"].append(f"metamask: {type(exc).__name__}: {exc}")
        print(f"  metamask  FAILED: {exc}")
    man["sources"]["metamask"] = mm

    man["coverage"] = {
        "arbitrum-one": "alchemy",
        "optimism": "covalent (erc20 yes/maybe + nft)",
        "base": "covalent (erc20 yes/maybe + nft)",
        "zksync-era": "alchemy (suspect capped)",
        "polygon-zkevm": "NONE -- no external label source; disclose as a limitation",
        "chain_independent": "metamask eth-phishing-detect domain blocklist",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"\n  manifest -> {manifest_path.relative_to(ROOT)}")
    if man["errors"]:
        print(f"  {len(man['errors'])} source(s) failed (recorded, not silently empty):", file=sys.stderr)
        for e in man["errors"]:
            print(f"    {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
