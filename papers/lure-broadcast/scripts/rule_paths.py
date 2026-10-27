"""What a detector verdict's two halves rest on (IEEE review R2.13; plan WP5.4, defect C5).

`metadata_lure.classify` calls a payload strong when it has a destination AND a call to action. Each
half can be satisfied in more than one way, and three ways satisfy the call to action without any
instruction outside the destination:
  * verb_in_url_only: every lexicon word lies inside a URL span (host or path), e.g.
    "bonus-claim.xyz", because word extraction runs over the whole normalised text;
  * telegram: a t.me handle, which supplies the destination as well (t.me is an aggregator host);
  * price: "$" followed by a digit.
A strong verdict whose call to action rests only on these paths is a "shortcut" through the
conjunction. Counting shortcuts quantifies how much weaker the rule is than its description.

The frozen detector's own regexes, TLD list and lexicon are used, and nothing here changes a verdict.
That module is hashed into the v2 annotation manifest and must not be edited, which is why this
logic lives here rather than there.

Pure functions; no I/O. Complexity O(len(text)).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402

CTA_PATHS = ("verb_outside_url", "verb_in_url_only", "telegram", "price")
DEST_PATHS = ("domain", "aggregator", "scheme")


def url_spans(text: str) -> list[tuple[int, int]]:
    """Spans of host-plus-path matches whose TLD is valid, as the detector reads destinations."""
    return [(m.start(), m.end()) for m in ml._URL_RE.finditer(text)
            if m.group("tld").lower() in ml.LURE_TLDS]


def without_urls(text: str) -> str:
    """The text with every URL span replaced by a space, so neighbouring words do not fuse."""
    out, last = [], 0
    for a, b in url_spans(text):
        out.append(text[last:a])
        out.append(" ")
        last = b
    out.append(text[last:])
    return "".join(out)


def cta_basis(verdict: ml.LureVerdict) -> frozenset[str]:
    """Which paths satisfy the call-to-action half. Empty when it is not satisfied."""
    basis: set[str] = set()
    if verdict.verbs:
        basis.add("verb_outside_url" if ml.extract_verbs(without_urls(verdict.normalised))
                  else "verb_in_url_only")
    if verdict.telegram_handles:
        basis.add("telegram")
    if verdict.has_price:
        basis.add("price")
    return frozenset(basis)


def destination_basis(verdict: ml.LureVerdict) -> frozenset[str]:
    """Which paths satisfy the destination half. Empty when it is not satisfied."""
    basis: set[str] = set()
    if verdict.domains:
        basis.add("domain")
    if verdict.aggregator_hosts:
        basis.add("aggregator")
    if verdict.has_scheme:
        basis.add("scheme")
    return frozenset(basis)


def is_shortcut(verdict: ml.LureVerdict) -> bool:
    """Strong, yet with no lexicon word outside the destination."""
    return verdict.strength == "strong" and "verb_outside_url" not in cta_basis(verdict)


def shortcut_kind(verdict: ml.LureVerdict) -> str | None:
    """A single label for reporting: 'genuine', or the shortcut path(s) joined with '+'."""
    if verdict.strength != "strong":
        return None
    basis = cta_basis(verdict)
    if "verb_outside_url" in basis:
        return "genuine"
    return "+".join(p for p in CTA_PATHS if p in basis)
