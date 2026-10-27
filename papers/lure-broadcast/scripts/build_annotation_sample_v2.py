"""Draw the v2 annotation sample: five chains, out-of-sample on Base, features judged separately.

Why a second sample (IEEE Access review R1.1/R2.1; plan WP2.3)
-------------------------------------------------------------
The v1 sample (`build_annotation_sample.py`, Base only) cannot validate the detector independently,
for three reasons:
  * its codebook was written alongside the rule, and both raters were authors;
  * after the labels were read, the detector was revised (IANA TLD list, plural stemming), so the
    recall measured on that sample is in-sample;
  * it covers one chain, while the detector is applied to five.

This draw addresses each of them:
  OUT-OF-SAMPLE. On Base, every contract in the v1 sample is excluded, so no row shown here can have
    influenced the detector. The other four chains were never annotated.
  FIVE CHAINS. The per-chain targets are declared below, before the draw.
  FEATURES, NOT THE RULE. Protocol v2 (docs/ANNOTATION_PROTOCOL_V2.md) asks for destination, call to
    action (any language) and judged purpose as separate fields. The rule's conjunction is never
    shown to raters and is derived from their answers afterwards.
  FROZEN DETECTOR. The SHA-256 of every file the verdict depends on is written into the manifest. A
    later change to any of them is therefore detectable, and a post-sample revision like v1's cannot
    happen unnoticed.

Blinding is stricter than v1. Raters see only row_id, name and symbol, plus the answer columns.
Chain and volume rank are withheld as well as every detector field, and rows from all chains are
shuffled together, so neither the stratum nor the chain can be read from the position of a row.

Inputs
------
analysis/config/token_metadata.json                          Base metadata (as for v1)
papers/lure-broadcast/data/token_metadata_<chain>.json       this paper's pinned sweep, other chains
results/<chain>_token_volumes.csv (or the paper's copy)      rank frame
papers/lure-broadcast/results/annotation/base_answer_key.json  the v1 rows, excluded on Base

Outputs (papers/lure-broadcast/results/annotation_v2/)
------------------------------------------------------
sample_blinded.csv     UTF-8 with BOM, so spreadsheet programs keep homoglyphs; GIVE TO RATERS
answer_key.json        strata, weights, detector verdicts per row; DO NOT GIVE TO RATERS
sample.manifest.json   seed, design, git commit, SHA-256 of the frozen detector and protocol

Failure modes
-------------
- Refuses to overwrite without --force: a redraw after labelling has begun invalidates the study.
- Raises if the allocation floors cannot fit the target, if a forbidden column would reach the
  blinded file, or if a pinned input is missing.
- Contracts with neither a name nor a symbol (unresolved, e.g. a reverted call) are excluded and
  counted, never shown as empty strings. Contracts absent from the rank frame cannot be weighted, so
  they are excluded and counted, as in v1.

Complexity: O(contracts) plus a streaming read of each rank file until every frame contract is found.
Seconds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402

OUT = PAPER / "results" / "annotation_v2"
PROTOCOL = PAPER / "docs" / "ANNOTATION_PROTOCOL_V2.md"
#: The protocol version the sample was DRAWN under, kept so that a re-run reproduces the committed
#: answer key byte for byte. Raters receive v2.1 (PREREG_ANNOTATION_V2.md §9, deviation 2). v2.1
#: changes the rater instructions only, never the rows, columns or weights. A re-run's manifest
#: will therefore hash the v2.1 text, while the committed draw manifest holds the v2.0 digest.
PROTOCOL_VERSION = "2.0"
V1_KEY = PAPER / "results" / "annotation" / "base_answer_key.json"

#: Rows per chain, declared before the draw. Base, Arbitrum and Optimism carry most of the flagged
#: contracts; zkSync has a large weak band; Polygon zkEVM has three flagged contracts in all.
#: Polygon zkEVM was registered at 30. Its eight strata need 38 rows under the floor, and `allocate`
#: refused the draw before writing anything. The target was raised to exactly the floor sum, keeping
#: the floor rule as registered (PREREG_ANNOTATION_V2.md §9, deviation 1).
TARGETS: dict[str, int] = {"base": 100, "arbitrum-one": 100, "optimism": 100,
                           "zksync-era": 70, "polygon-zkevm": 38}
#: Every stratum contributes at least this many rows, or all of its rows if it has fewer.
FLOOR = 5
#: Pre-declared seed (the date the draw was designed).
DEFAULT_SEED = 20261027

RANK_BANDS = [(0, 200), (200, 600), (600, 1200)]
#: How far into a rank file to read before giving up on the remaining frame contracts.
RANK_SCAN_LIMIT = 200_000

#: What a rater may see. Everything else is a potential leak.
BLINDED_COLUMNS = ["row_id", "name", "symbol", "destination", "call_to_action", "cta_language",
                   "purpose", "notes"]
FORBIDDEN_COLUMNS = {"chain", "volume_rank", "strength", "stratum", "is_lure", "campaign_key",
                     "domains", "verbs", "reasons", "address", "contract", "detector"}

#: Every file the verdict depends on, plus the protocol, hashed into the manifest.
FROZEN_INPUTS = [
    ROOT / "analysis" / "src" / "metadata_lure.py",
    ROOT / "analysis" / "src" / "confusables.py",
    PAPER / "data" / "tlds" / "tlds-alpha-by-domain.txt",
    PROTOCOL,
]


def metadata_path(chain: str) -> Path:
    if chain == "base":
        return ROOT / "analysis" / "config" / "token_metadata.json"
    return PAPER / "data" / f"token_metadata_{chain}.json"


def rank_path(chain: str) -> Path:
    shared = ROOT / "results" / f"{chain}_token_volumes.csv"
    return shared if shared.exists() else PAPER / "results" / f"{chain}_token_volumes.csv"


def _shown(path: Path) -> str:
    """Repository-relative path for messages, or the absolute path when it lies outside the repo.

    A bare `relative_to` raises on an outside path. In an error message, that replaces the refusal
    with a traceback.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def band_of(rank: int) -> str:
    """Volume-rank band, identical to v1 so the two samples stratify alike."""
    for lo, hi in RANK_BANDS:
        if lo <= rank < hi:
            return f"{lo}-{hi}"
    return f"{RANK_BANDS[-1][1]}+"


def is_unresolved(entry: Mapping) -> bool:
    """True when the contract reported no strings at all, so there is nothing to judge."""
    return not (str(entry.get("name") or "").strip() or str(entry.get("symbol") or "").strip())


def detector_features(verdict: ml.LureVerdict) -> dict:
    """The two halves of the rule, as `metadata_lure.classify` computes them, for the key only."""
    return {
        "destination": bool(verdict.domains) or bool(verdict.aggregator_hosts) or verdict.has_scheme,
        "call_to_action": bool(verdict.verbs) or bool(verdict.telegram_handles) or verdict.has_price,
        "domains": list(verdict.domains),
        "aggregator_hosts": list(verdict.aggregator_hosts),
        "benign_domains": list(verdict.benign_domains),
        "verbs": list(verdict.verbs),
        "telegram_handles": list(verdict.telegram_handles),
        "has_scheme": verdict.has_scheme,
        "has_price": verdict.has_price,
    }


def read_ranks(path: Path, wanted: set[str], limit: int = RANK_SCAN_LIMIT) -> dict[str, int]:
    """Zero-based row index of each wanted address in a rank file (sorted by transfer count).

    Streams the file and stops once every wanted address is found or `limit` rows are read. The frame
    is the volume head, so on the 24.6M-row Base file this reads about 1,200 rows rather than all.
    """
    found: dict[str, int] = {}
    with path.open() as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            a = row["token"].lower()
            if a in wanted and a not in found:
                found[a] = i
                if len(found) == len(wanted):
                    break
            if i >= limit:
                break
    return found


def build_population(metadata: Mapping[str, Mapping], ranks: Mapping[str, int],
                     exclude: Iterable[str] = ()) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Classify every frame contract and group it into strength x rank-band strata.

    Returns the strata and a count of each exclusion, so no contract silently disappears.
    """
    excluded_set = {a.lower() for a in exclude}
    strata: dict[str, list[dict]] = {}
    dropped = {"v1_sample": 0, "unresolved": 0, "not_in_frame": 0}
    for addr, entry in metadata.items():
        a = addr.lower()
        if a in excluded_set:
            dropped["v1_sample"] += 1
            continue
        if not isinstance(entry, Mapping) or is_unresolved(entry):
            dropped["unresolved"] += 1
            continue
        r = ranks.get(a)
        if r is None:
            dropped["not_in_frame"] += 1
            continue
        v = ml.classify(entry.get("name"), entry.get("symbol"))
        key = f"{v.strength}|{band_of(r)}"
        strata.setdefault(key, []).append({
            "address": a, "name": entry.get("name") or "", "symbol": entry.get("symbol") or "",
            "volume_rank": r, "strength": v.strength, "stratum": key,
            "campaign_key": ml.campaign_key(v) or "", "detector": detector_features(v),
        })
    return strata, dropped


def allocate(sizes: Mapping[str, int], n: int, floor: int = FLOOR) -> dict[str, int]:
    """Split `n` rows across strata: a floor each, the rest in proportion to sqrt(N_h), capped at N_h.

    The square root is a compromise. Proportional allocation would starve the small weak and strong
    strata, which carry the cases of interest, while equal allocation would over-weight them. The
    floor guarantees every stratum is represented, so recall stays estimable. The cap means a small
    stratum is taken whole rather than producing a shortfall.

    Deterministic: ties go to the lexicographically smaller stratum name. The result always sums to
    min(n, sum(sizes)).
    Raises ValueError on an empty or non-positive stratum, or when the floors alone exceed `n`.
    """
    if not sizes:
        raise ValueError("allocate: no strata")
    if any(v <= 0 for v in sizes.values()):
        raise ValueError(f"allocate: every stratum must be non-empty, got {dict(sizes)}")
    n = min(n, sum(sizes.values()))
    alloc = {k: min(floor, v) for k, v in sizes.items()}
    if sum(alloc.values()) > n:
        raise ValueError(f"allocate: floors need {sum(alloc.values())} rows but the target is {n}")
    remaining = n - sum(alloc.values())
    while remaining > 0:
        open_ = sorted(k for k in sizes if alloc[k] < sizes[k])
        weight = {k: math.sqrt(sizes[k]) for k in open_}
        total = sum(weight.values())
        quota = {k: remaining * weight[k] / total for k in open_}
        give = {k: min(int(quota[k]), sizes[k] - alloc[k]) for k in open_}
        if sum(give.values()) == 0:
            # Every quota is below one: hand a single row to the largest fractional share.
            best = sorted(open_, key=lambda k: (-(quota[k] - int(quota[k])), k))[0]
            give = {best: 1}
        for k, g in give.items():
            alloc[k] += g
            remaining -= g
    return alloc


def draw(strata: Mapping[str, list[dict]], alloc: Mapping[str, int], rng: random.Random) -> list[dict]:
    """Simple random sample without replacement within each stratum.

    Rows are sorted by address before sampling, so the draw depends on the seed alone and not on
    dictionary order.
    """
    chosen: list[dict] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda r: r["address"])
        chosen.extend(rng.sample(rows, alloc[key]))
    return chosen


#: A cell starting with one of these is evaluated as a formula by Excel, LibreOffice and Google
#: Sheets. Token names and symbols are written by whoever deployed the contract, so a sheet built
#: from them can carry a live formula, such as a HYPERLINK, to the rater (CSV injection).
FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def blinded_rows(chosen: list[dict]) -> list[dict]:
    """What raters see: identifier, the two strings and empty answer columns. Nothing else.

    Raises rather than escaping when a string would be read as a formula. Escaping (for example a
    leading apostrophe) would change the very text the rater is asked to judge. A dangerous sheet
    must therefore stop the draw, not be repaired silently.
    """
    leaks = FORBIDDEN_COLUMNS & set(BLINDED_COLUMNS)
    if leaks:
        raise ValueError(f"BLINDED_COLUMNS leaks {sorted(leaks)}")
    for r in chosen:
        for field_name in ("name", "symbol"):
            if str(r[field_name]).startswith(FORMULA_TRIGGERS):
                raise ValueError(f"{r['row_id']} {field_name} starts with a spreadsheet formula "
                                 f"trigger ({str(r[field_name])[:1]!r}); refusing to write a sheet "
                                 f"that would execute it on the rater's machine")
    return [{"row_id": r["row_id"], "name": r["name"], "symbol": r["symbol"], "destination": "",
             "call_to_action": "", "cta_language": "", "purpose": "", "notes": ""} for r in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    blinded_path = OUT / "sample_blinded.csv"
    if blinded_path.exists() and not args.force:
        raise SystemExit(f"{_shown(blinded_path)} exists; pass --force. A redraw after "
                         f"labelling has started invalidates the study.")
    for p in FROZEN_INPUTS + [V1_KEY]:
        if not p.exists():
            raise SystemExit(f"pinned input missing: {_shown(p)}")

    v1_addresses = [row["address"] for row in json.loads(V1_KEY.read_text())["rows"].values()]
    rng = random.Random(args.seed)
    chosen: list[dict] = []
    per_chain: dict[str, dict] = {}
    for chain in sorted(TARGETS):
        md_all = json.loads(metadata_path(chain).read_text())
        md = md_all.get(chain, md_all) if chain == "base" else md_all
        ranks = read_ranks(rank_path(chain), {a.lower() for a in md})
        strata, dropped = build_population(md, ranks, v1_addresses if chain == "base" else ())
        alloc = allocate({k: len(v) for k, v in strata.items()}, TARGETS[chain])
        rows = draw(strata, alloc, rng)
        for r in rows:
            r["chain"] = chain
        chosen.extend(rows)
        per_chain[chain] = {
            "target": TARGETS[chain],
            "population_by_stratum": {k: len(v) for k, v in sorted(strata.items())},
            "sampled_by_stratum": dict(sorted(alloc.items())),
            "population_total": sum(len(v) for v in strata.values()),
            "excluded": dropped,
            "metadata_source": _shown(metadata_path(chain)),
            "rank_source": _shown(rank_path(chain)),
        }

    rng.shuffle(chosen)                       # across chains: position reveals neither chain nor stratum
    for i, row in enumerate(chosen, start=1):
        row["row_id"] = f"r{i:04d}"

    OUT.mkdir(parents=True, exist_ok=True)
    with blinded_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=BLINDED_COLUMNS)
        w.writeheader()
        w.writerows(blinded_rows(chosen))

    key = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": args.seed,
        "floor": FLOOR,
        "allocation": "floor per stratum, remainder proportional to sqrt(N_h), capped at N_h",
        "chains": per_chain,
        "sample_size": len(chosen),
        "rows": {r["row_id"]: {k: r[k] for k in ("chain", "address", "stratum", "strength",
                                                 "volume_rank", "campaign_key", "detector")}
                 for r in chosen},
        "note": ("DO NOT SHARE WITH RATERS. On Base the population excludes every v1 row, so the "
                 "weights describe the out-of-sample remainder. Strata use the current detector "
                 "bands, which the manifest hashes."),
    }
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=1, sort_keys=True,
                                                    ensure_ascii=False) + "\n")
    (OUT / "sample.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/build_annotation_sample_v2.py",
        "git_commit": git_commit(),
        "seed": args.seed,
        "protocol": f"{_shown(PROTOCOL)} v{PROTOCOL_VERSION}",
        "frozen_inputs_sha256": {_shown(p): sha256(p) for p in FROZEN_INPUTS},
        "blinded_sha256": sha256(blinded_path),
        "blinded_columns": BLINDED_COLUMNS,
        "targets": TARGETS,
    }, indent=1) + "\n")

    print(f"  sample: {len(chosen)} rows over {len(TARGETS)} chains")
    for chain, info in per_chain.items():
        print(f"  {chain:14s} population {info['population_total']:5d}  excluded {info['excluded']}")
        for k, n in info["sampled_by_stratum"].items():
            print(f"      {k:22s} N_h {info['population_by_stratum'][k]:5d}  n_h {n:3d}")
    print(f"  -> {_shown(blinded_path)}  (GIVE TO RATERS)")
    print(f"  -> {_shown((OUT / 'answer_key.json'))}  (DO NOT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
