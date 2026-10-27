"""How far apart would registry coverage have to be before the observed contrast reverses?

Purpose
-------
The activity measure counts an address as active only if it sent a non-zero transfer of a
registry-listed token, so it observes a fraction of true activity rather than all of it. The
manuscript previously argued that applying the same rule to the recipients and to the comparison
group left the contrast between them unaffected. That inference is wrong, and a referee was right to
reject it: an identical measurement RULE does not imply an identical CAPTURE RATE. Writing c for the
probability that a truly active address is seen to be active,

    q1 = c1 * p1,    q0 = c0 * p0,

the observed inequality q1 > q0 implies the true inequality p1 > p0 only when c1 / c0 < q1 / q0.

Rather than assume the capture rates are equal, this script reports the exact point at which the
conclusion would fail. The observed direction is preserved unless

    c1 / c0  >=  q1 / q0      (when q1 > q0),   or
    c1 / c0  <=  q1 / q0      (when q1 < q0),

so the ratio q1 / q0 is itself the reversal threshold, and it needs no assumption at all: it is a
statement about how differently the registry would have to behave in the two groups. The
interval-aware version uses the confidence limits that make reversal easiest, which gives the
conservative threshold a reader should quote.

The capture rates are not themselves estimated, and no attempt is made to do so. The obvious proxy,
the ratio of registry-observed to contract-asserted activity, is unusable here: the asserted
quantity is exactly the one this paper shows to be fabricable, and the recipient group is by
construction the population that fabricating contracts name. Reporting it as a coverage estimate
would reintroduce the observable the analysis exists to avoid.

Inputs   papers/lure-broadcast/results/recipient_activity_<chain>.json
Outputs  papers/lure-broadcast/results/recipient_coverage_sensitivity.json (+ manifest)

Failure modes
-------------
- A chain whose control rate is zero yields an undefined ratio and is reported as null, never as
  infinity or as a large finite number.
- A chain whose intervals overlap has no observed direction to preserve; it is labelled as such
  rather than being given a threshold that would imply one.

Complexity: arithmetic over the saved proportions. No corpus pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "papers" / "lure-broadcast" / "results"
CHAINS = ["base", "arbitrum-one", "optimism"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def threshold(q1: float, q0: float) -> float | None:
    """Reversal threshold on the capture ratio c1 / c0. None when q0 is zero."""
    if q0 <= 0:
        return None
    return q1 / q0


def analyse(doc: dict) -> dict:
    act, ctl = doc["active"], doc["control"]["active"]
    q1, q0 = act["estimate"], ctl["estimate"]
    if q1 is None or q0 is None:
        return {"direction": "undefined", "threshold": None}

    separated = act["ci95_lo"] > ctl["ci95_hi"] or act["ci95_hi"] < ctl["ci95_lo"]
    direction = "above" if q1 > q0 else ("below" if q1 < q0 else "equal")

    # The conservative threshold uses the interval limits that make reversal easiest: for an
    # observed excess, the smallest plausible q1 against the largest plausible q0.
    if direction == "above":
        conservative = threshold(act["ci95_lo"], ctl["ci95_hi"])
    elif direction == "below":
        conservative = threshold(act["ci95_hi"], ctl["ci95_lo"])
    else:
        conservative = None

    return {
        "q_named": q1, "q_control": q0,
        "direction": direction,
        "intervals_separated": bool(separated),
        "threshold_point": threshold(q1, q0),
        "threshold_conservative": conservative,
        "reading": (
            "the observed direction reverses only if the registry captures activity in the named "
            "group at least this many times more completely than in the comparison group"
            if direction == "above" else
            "the observed direction reverses only if the registry captures activity in the named "
            "group at most this fraction as completely as in the comparison group"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = RES / "recipient_coverage_sensitivity.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    per_chain = {}
    for c in CHAINS:
        f = RES / f"recipient_activity_{c}.json"
        if not f.exists():
            raise SystemExit(f"missing {f.name}; run recipient_activity.py for every chain the "
                             f"manuscript reports rather than bounding a subset of them")
        per_chain[c] = analyse(json.loads(f.read_text()))

    doc = {
        "question": ("how different would registry capture have to be between the named and "
                     "comparison groups before the observed contrast reversed?"),
        "model": ("q = c * p, with q the observed rate, p the true activity rate and c the "
                  "probability that a truly active address transacts in a registry-listed token. "
                  "The observed direction is preserved unless c_named / c_control crosses q_named "
                  "/ q_control, so that ratio is the exact reversal threshold and requires no "
                  "assumption about the capture rates themselves."),
        "not_estimated": ("the capture rates are not estimated. The available proxy, the ratio of "
                          "registry-observed to contract-asserted activity, is unusable: the "
                          "asserted quantity is the fabricable one, and the named group is by "
                          "construction the population that fabricating contracts name."),
        "per_chain": per_chain,
    }
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (RES / "recipient_coverage_sensitivity.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/coverage_sensitivity.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    for c, v in per_chain.items():
        print(f"  {c:14s} {v['direction']:>5s}  point {v['threshold_point']:.3f}  "
              f"conservative {v['threshold_conservative']:.3f}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
