"""Adversarial and evasion evaluation of the detector (reviewer item 6).

Why this is needed and what it can and cannot show
--------------------------------------------------
The annotated reference set is stratified by the detector's own band, so it measures agreement with
the operational definition over payloads that exist in the corpus. It cannot say how the rule behaves
against a payload built to defeat it, and it cannot supply the external validity that only
independent annotation would give. This suite addresses the first gap and not the second.

Each case carries an expectation DECLARED WITH THE CASE, not after seeing the outcome, and the suite
deliberately contains cases we expect the detector to FAIL. A suite that only its author's detector
passes measures nothing; the value is in locating the boundary.

Both directions are tested. Evasion cases are positives that a working detector should still flag.
Decoy cases are benign payloads engineered to look like the positive class -- an imperative verb in a
legitimate name, a version string resembling a host, a brand with a dot -- which a detector must NOT
flag. Reporting only the first would reward a rule that flags everything.

Interpretation of a miss
------------------------
A miss is reported, not patched. The English-only call-to-action lexicon is a known and deliberate
limitation, and the multilingual cases exist to quantify it rather than to be fixed by adding words
until the suite passes; doing that would make the suite a training set and destroy its value.

Outputs
-------
papers/lure-broadcast/results/adversarial_eval.json (+ .manifest.json)

Failure modes
-------------
- A case whose expectation is not met is recorded as a discrepancy and does NOT raise: discrepancies
  are the finding.
- If every decoy is flagged the run raises, because a detector that flags all decoys has no
  discriminating power and the evasion results would be meaningless.

Complexity: trivial; one classification per case.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

#: (category, name, symbol, expect_flagged, note). Expectations are part of the case definition.
CASES: list[tuple[str, str, str, bool, str]] = [
    # --- baseline: the plain positive the rule is built for -------------------------------------
    ("baseline", "USDC Reward", "claim at bonus-zk.com", True, "plain domain plus imperative"),
    ("baseline", "Airdrop", "visit https://claim-arb.io now", True, "explicit scheme"),

    # --- character-level evasion ----------------------------------------------------------------
    ("homoglyph", "Reward", "claim at www.fomoх.club", True, "Cyrillic kha inside the host"),
    ("homoglyph", "Bonus", "visit www.fomox.сlub today", True, "Cyrillic es inside the suffix"),
    ("invisible", "Reward", "claim at bonus​zk.com", True, "zero-width space inside the host"),
    ("invisible", "Reward", "claim at bonus­zk.com", True, "soft hyphen inside the host"),
    ("dot_obfuscation", "Reward", "claim at bonus-zk[.]com", True, "bracketed dot"),
    ("dot_obfuscation", "Reward", "claim at bonus-zk(dot)com", True, "spelled dot"),
    ("dot_obfuscation", "Reward", "claim at bonus-zk DOT com", True, "spaced and capitalised dot"),

    # --- structural evasion ---------------------------------------------------------------------
    ("shortener", "Reward", "claim via t.ly/optimism-drop", True, "URL shortener with a slug"),
    ("shortener", "Reward", "claim via bit.ly/3xKq9", True, "opaque shortener slug"),
    ("delimiter", "Reward", "claim at bonus-zk.com", True, "non-breaking spaces"),
    ("delimiter", "Reward |", "| claim | bonus-zk.com |", True, "pipe delimiters"),
    ("scheme_obfuscation", "Reward", "claim at hxxps://bonus-zk.com", True, "defanged scheme"),

    # --- language. A NEUTRAL host is essential here: an earlier version of this suite used
    # `bonus-zk.com`, whose own name contains a lexicon word, so every multilingual case passed on
    # the domain rather than on the call to action and the test measured nothing.
    ("multilingual", "Recompensa", "reclama en zkp-portal.xyz", True, "Spanish imperative"),
    ("multilingual", "Belohnung", "jetzt einloesen zkp-portal.xyz", True, "German imperative"),
    ("multilingual", "奖励", "领取 zkp-portal.xyz", True, "Chinese imperative"),

    # --- the conjunction is not always a conjunction. The verb lexicon is matched against the whole
    # normalised payload, the extracted host included, so a host whose own name contains a lexicon
    # word satisfies the call-to-action test by itself. These cases quantify that; the expectation
    # records what a reader would assume the rule does, so a disagreement is the finding.
    ("domain_self_satisfies", "", "bonus-zk.com", False, "bare host, no call to action present"),
    ("domain_self_satisfies", "", "airdrop-portal.xyz", False, "bare host, no call to action present"),
    ("domain_self_satisfies", "", "zkp-portal.xyz", False, "bare neutral host, correctly not strong"),

    # --- decoys: benign payloads shaped like the positive class -----------------------------------
    ("decoy_verb", "Claimable Yield Token", "CYT", False, "imperative-like word, no destination"),
    ("decoy_verb", "Free Float Index", "FFI", False, "'free' in a legitimate name"),
    ("decoy_version", "Uniswap V2", "UNI-V2", False, "version string, not a host"),
    ("decoy_version", "Curve.fi DAI/USDC", "3CRV", False, "legitimate brand containing a dot"),
    ("decoy_domain", "Aave interest bearing USDC", "aUSDC", False, "no destination, no call to action"),
    ("decoy_domain", "Wrapped Ether", "WETH", False, "plain legitimate token"),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "adversarial_eval.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    rows = []
    for cat, name, sym, expect, note in CASES:
        v = ml.classify(name, sym)
        flagged = v.strength == "strong"
        rows.append({"category": cat, "name": name, "symbol": sym, "note": note,
                     "expected_flagged": expect, "flagged": flagged,
                     "strength": v.strength, "campaign": ml.campaign_key(v),
                     "agrees": flagged == expect})

    evasion = [r for r in rows if r["expected_flagged"]]
    decoys = [r for r in rows if not r["expected_flagged"]]
    caught = sum(1 for r in evasion if r["flagged"])
    false_alarms = sum(1 for r in decoys if r["flagged"])

    if decoys and false_alarms == len(decoys):
        raise SystemExit("every decoy was flagged; the detector has no discriminating power on this "
                         "suite and the evasion figures would be meaningless")

    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "agree": 0})
        c["n"] += 1
        c["agree"] += int(r["agrees"])

    misses = [r for r in evasion if not r["flagged"]]
    doc = {
        "purpose": ("locate the detector's boundary against payloads constructed to defeat it; this "
                    "tests generalisation beyond the annotation task and does NOT supply the "
                    "independent confirmation of intent that only external annotation would give"),
        "expectations": "declared with each case before the run, not fitted to the outcome",
        "evasion": {"n": len(evasion), "flagged": caught, "detection_rate": caught / len(evasion)},
        "decoys": {"n": len(decoys), "false_alarms": false_alarms,
                   "false_alarm_rate": false_alarms / len(decoys) if decoys else None},
        "by_category": {k: {**v, "agreement_rate": v["agree"] / v["n"]} for k, v in sorted(by_cat.items())},
        "misses": [{"category": r["category"], "symbol": r["symbol"], "note": r["note"],
                    "strength": r["strength"]} for r in misses],
        "miss_note": ("misses are reported, not patched: extending the lexicon until the suite "
                      "passes would convert it into a training set and destroy its diagnostic value"),
        "cases": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "adversarial_eval.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/adversarial_eval.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    print(f"  evasion cases {caught}/{len(evasion)} flagged "
          f"({caught/len(evasion)*100:.1f}%)")
    print(f"  decoys        {false_alarms}/{len(decoys)} falsely flagged")
    for k, v in sorted(by_cat.items()):
        print(f"    {k:20s} {v['agree']}/{v['n']} as expected")
    if misses:
        print("  MISSES (reported, not patched):")
        for r in misses:
            print(f"    [{r['category']}] {r['symbol'][:52]!r} -> {r['strength']}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
