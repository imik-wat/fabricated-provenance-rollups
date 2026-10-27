"""How the detector rule actually behaves on the full frame, per chain (IEEE R1.2, R2.12, R2.13; WP5, C5, C11).

Four measurements over the detector population of each chain (the cached metadata the paper's counts
come from), with no network and no change to any verdict:

  SHORTCUTS (5.4). How many strong contracts satisfy the call-to-action half only through a lexicon
    word inside a URL, a Telegram handle or a price pattern (rule_paths.py). Also what share of the
    channel's transfer volume they emit, from the per-contract counts in forgery_chain_<chain>.json.
  ALLOWLIST OFF (5.5). The frozen classifier with BENIGN_DOMAINS emptied, which is the list the paper
    failed to disclose. It reports which contracts would become strong and through which allowlisted
    domains. The module attribute is swapped for the duration of the call and restored in a finally
    block. The file itself is hashed for the v2 study and is never edited.
  SCRIPT CENSUS (5.1). Payloads containing letters of each Unicode script, by character name. Also
    how many Cyrillic-bearing payloads fold entirely to Latin under the confusable skeleton, i.e. use
    Cyrillic as lookalikes rather than as Russian.
  LEXICON BOUND (5.2). The rule needs a destination, so within the frame the English-only lexicon can
    cause a miss only on a payload that has a destination but no detected call to action: the
    destination-only weak payloads. That count is a hard upper bound on lexicon-induced misses among
    examined contracts. Those carrying non-Latin letters are listed for blind review.

Inputs:  analysis/config/token_metadata.json (Base), papers/lure-broadcast/data/token_metadata_<chain>.json
         papers/lure-broadcast/results/forgery_chain_<chain>.json (transfers per flagged contract)
Output:  papers/lure-broadcast/results/rule_interaction.json (+ .manifest.json)
Complexity: O(contracts x payload length). Seconds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml  # noqa: E402
import rule_paths as rp  # noqa: E402
from build_annotation_sample_v2 import is_unresolved, metadata_path  # noqa: E402

CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]
OUT = PAPER / "results" / "rule_interaction.json"

#: Unicode character-name markers, checked in order. Decorative Latin (fullwidth, circled,
#: mathematical alphanumerics) is Latin for this purpose, since it carries no other language.
SCRIPT_MARKERS = [
    ("latin", ("LATIN", "MATHEMATICAL")),
    ("cjk", ("CJK", "HIRAGANA", "KATAKANA", "IDEOGRAPH", "BOPOMOFO")),
    ("hangul", ("HANGUL",)),
    ("cyrillic", ("CYRILLIC",)),
    ("arabic", ("ARABIC",)),
    ("thai", ("THAI",)),
    ("greek", ("GREEK",)),
    ("devanagari", ("DEVANAGARI",)),
    ("hebrew", ("HEBREW",)),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def script_of(ch: str) -> str | None:
    """The script of a letter, or None for anything that is not a letter."""
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    for script, markers in SCRIPT_MARKERS:
        if any(m in name for m in markers):
            return script
    return "other"


def scripts_in(text: str) -> frozenset[str]:
    return frozenset(s for s in map(script_of, text) if s is not None)


def lookalike_only(text: str) -> bool:
    """True if the text has non-Latin letters and every one of them folds to Latin under the skeleton.

    Such a payload uses another script to imitate Latin (Lisu "ꓴꓢꓓ" reads USD) rather than to write
    another language.
    """
    return bool(scripts_in(text) - {"latin"}) and not (scripts_in(ml.skeleton(text)) - {"latin"})


def classify_all(md: dict) -> dict[str, ml.LureVerdict]:
    return {a.lower(): ml.classify(e.get("name"), e.get("symbol"))
            for a, e in md.items() if isinstance(e, dict) and not is_unresolved(e)}


def with_allowlist_disabled(md: dict) -> dict[str, ml.LureVerdict]:
    """Classify with BENIGN_DOMAINS emptied for the duration of the call, then restore it."""
    saved = ml.BENIGN_DOMAINS
    try:
        ml.BENIGN_DOMAINS = frozenset()
        return classify_all(md)
    finally:
        ml.BENIGN_DOMAINS = saved


def summarise_chain(md: dict, transfers: dict[str, int]) -> dict:
    verdicts = classify_all(md)
    payload = {a.lower(): " ".join(x for x in (e.get("name"), e.get("symbol")) if x)
               for a, e in md.items() if isinstance(e, dict) and not is_unresolved(e)}
    bands = Counter(v.strength for v in verdicts.values())
    strong = {a: v for a, v in verdicts.items() if v.strength == "strong"}

    kinds = Counter(rp.shortcut_kind(v) for v in strong.values())
    vol_by_kind: Counter = Counter()
    missing_volume = []
    for a, v in strong.items():
        if a in transfers:
            vol_by_kind[rp.shortcut_kind(v)] += transfers[a]
        else:
            missing_volume.append(a)
    total_vol = sum(vol_by_kind.values())
    shortcut_vol = sum(n for k, n in vol_by_kind.items() if k != "genuine")
    campaigns: dict[str, list[bool]] = {}
    for v in strong.values():
        key = ml.campaign_key(v)
        if key:
            campaigns.setdefault(key, []).append(rp.is_shortcut(v))

    off = with_allowlist_disabled(md)
    added = sorted(a for a, v in off.items() if v.strength == "strong" and a not in strong)
    removed = sorted(a for a in strong if off.get(a) is None or off[a].strength != "strong")
    via = Counter(d for a in added for d in verdicts[a].benign_domains)

    census = Counter()
    homoglyph_only = 0
    n_lookalike = 0
    for a, text in payload.items():
        found = scripts_in(text)
        for s in found:
            census[s] += 1
        if found - {"latin"}:
            census["any_non_latin"] += 1
            if lookalike_only(text):
                n_lookalike += 1
        if "cyrillic" in found and "cyrillic" not in scripts_in(ml.skeleton(text)):
            homoglyph_only += 1

    feats = {a: (bool(rp.destination_basis(v)), bool(rp.cta_basis(v))) for a, v in verdicts.items()}
    dest_only = sorted(a for a, (d, c) in feats.items() if d and not c)
    cta_only = sum(1 for d, c in feats.values() if c and not d)
    non_latin_dest_only = [a for a in dest_only if scripts_in(payload[a]) - {"latin"}]
    return {
        "population": len(verdicts),
        "unresolved": len(md) - len(verdicts),
        "bands": dict(sorted(bands.items())),
        "strong_by_cta_path": dict(sorted(kinds.items())),
        "shortcut_contracts": sum(n for k, n in kinds.items() if k != "genuine"),
        "transfers_by_cta_path": dict(sorted(vol_by_kind.items())),
        "shortcut_transfer_share": shortcut_vol / total_vol if total_vol else None,
        "strong_without_transfer_count": missing_volume,
        "campaigns": len(campaigns),
        "campaigns_all_shortcut": sum(1 for flags in campaigns.values() if all(flags)),
        "allowlist_off": {"strong": len(strong) - len(removed) + len(added), "added": added,
                          "removed": removed, "added_via_domain": dict(sorted(via.items()))},
        "script_census": dict(sorted(census.items())),
        "cyrillic_homoglyph_only": homoglyph_only,
        "non_latin_lookalike_only": n_lookalike,
        "lexicon_bound": {
            "destination_only_weak": len(dest_only),
            "destination_only_weak_addresses": dest_only,
            "destination_only_weak_with_non_latin": len(non_latin_dest_only),
            # Rows whose non-Latin letters all imitate Latin: a miss among them is an English miss.
            "destination_only_weak_non_latin_lookalike_only": sum(
                1 for a in non_latin_dest_only if lookalike_only(payload[a])),
            "non_latin_rows_for_review": [{"address": a, "name": md_entry(md, a)["name"],
                                           "symbol": md_entry(md, a)["symbol"]}
                                          for a in non_latin_dest_only],
            "cta_only_weak": cta_only,
            "note": ("within the examined frame a lexicon miss requires a destination, so "
                     "destination_only_weak bounds lexicon-induced misses from above; it is not "
                     "an estimate of them"),
        },
    }


def md_entry(md: dict, address: str) -> dict:
    for a, e in md.items():
        if a.lower() == address:
            return {"name": e.get("name") or "", "symbol": e.get("symbol") or ""}
    raise KeyError(address)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists; pass --force")
    result = {}
    for chain in CHAINS:
        raw = json.loads(metadata_path(chain).read_text())
        md = raw[chain] if chain == "base" else raw
        fc = json.loads((PAPER / "results" / f"forgery_chain_{chain}.json").read_text())
        transfers = {a.lower(): int(n) for a, n in fc["lure_transfers_by_contract"].items()}
        result[chain] = summarise_chain(md, transfers)
        r = result[chain]
        print(f"  {chain:14s} strong {r['bands'].get('strong', 0):4d}  shortcuts {r['shortcut_contracts']:3d}"
              f"  vol share {r['shortcut_transfer_share'] if r['shortcut_transfer_share'] is None else round(r['shortcut_transfer_share'], 4)}"
              f"  allowlist-off +{len(r['allowlist_off']['added'])}"
              f"  dest-only weak {r['lexicon_bound']['destination_only_weak']}"
              f" (non-Latin {r['lexicon_bound']['destination_only_weak_with_non_latin']})")
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/rule_interaction.py", "git_commit": git_commit(),
    }, indent=1) + "\n")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
