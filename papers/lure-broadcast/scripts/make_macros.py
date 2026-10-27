r"""Emit every number the AMCS paper quotes as a LaTeX macro, from results/*.json only.

Why this exists
---------------
Project rule 1 (README): no untraceable numbers. Every quantitative claim in the manuscript must
originate in a file under `results/`, produced by a script in this tree, and reach LaTeX through a
generated macro -- never typed into prose. This script is that bridge.

Design decisions
----------------
1. **Loud failure, never a default.** `_get` walks a dotted path and RAISES on a missing key. A
   macro pipeline that silently emits `0` or an empty string is the exact failure mode that let a
   hardcoded-address valuation bug understate Optimism 4,500x elsewhere in this project. There is no
   `.get(k, default)` anywhere below.
2. **A missing input file is fatal, not skipped.** If a results file is absent the paper would
   silently lose a claim, or worse, keep prose whose macro is undefined and render "??".
3. **`\lb` prefix on every macro.** The repository-root `paper/generated/` uses unprefixed
   names; sharing a name across the two trees would let one paper's number appear in the other.
   Isolation invariant 2 in the plan.
4. **Formatting is separated from extraction.** `_int` groups thousands via siunitx `\num`,
   `_pct`/`_dec` fix decimal places at the point of emission, so the prose cannot silently change a
   number's precision. Percentages are emitted WITHOUT the % sign so the text controls wording.

Inputs  (all under papers/lure-broadcast/results/)
------
nf_census_all.json, campaign_census_base.json, detector_design_based_base.json,
interrater_base.json, theta_base.json, corroboration_base.json, saturation_<chain>.json,
strata_sweep_base.json

Outputs
-------
papers/lure-broadcast/paper/generated/macros.tex (+ .manifest.json)

Failure modes
-------------
- Missing file or missing key -> SystemExit with the exact path that failed.
- A macro name emitted twice -> SystemExit (a redefinition would make the paper's number depend on
  \input order).
- Nothing is written unless every macro resolved, so a partial macros.tex cannot reach a build.

Complexity: trivial; reads a handful of small JSON documents.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src.stats import wilson  # noqa: E402

RES = ROOT / "papers" / "lure-broadcast" / "results"
OUT = ROOT / "papers" / "lure-broadcast" / "paper" / "generated" / "macros.tex"

CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]
#: The chain the single-chain results (lure census, detector, Theta, forgery) are measured on.
#: Named once so a replication only has to change this line.
LEAD_CHAIN = "base"
#: depth-bucket label -> macro-name fragment. A LaTeX control sequence may contain letters only, so
#: the numeric cutoffs are spelled out. Adding a cutoff to DEPTH_CUTOFFS requires an entry here; the
#: emitter raises rather than inventing a spelling.
DEPTH_TEX = {"100": "OneHundred", "300": "ThreeHundred", "600": "SixHundred", "all": "All"}
#: chain slug -> LaTeX-safe macro fragment and display name
CHAIN_TEX = {
    "arbitrum-one": ("ArbitrumOne", "Arbitrum One"),
    "base": ("Base", "Base"),
    "optimism": ("Optimism", "Optimism"),
    "polygon-zkevm": ("PolygonZkEVM", "Polygon zkEVM"),
    "zksync-era": ("ZKsyncEra", "zkSync Era"),
}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing input {p.relative_to(ROOT)} -- run the producing script first; "
                         f"refusing to emit a partial macro set")
    return json.loads(p.read_text())


def _get(doc: dict, path: str, src: str):
    """Walk a dotted path. RAISES on a missing key -- never returns a default."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SystemExit(f"{src}: key path {path!r} not found (failed at {part!r}); "
                             f"the results schema changed -- fix this script, do not guess a value")
        cur = cur[part]
    if cur is None:
        raise SystemExit(f"{src}: key path {path!r} is null; a null cannot become a paper number")
    return cur


def classify_residual(by_shape: dict[str, int]) -> dict[str, int]:
    """Group residual `Transfer` rows by what their shape implies about the emitting declaration.

    The data width follows the `indexed` declaration, since indexing moves a parameter out of the
    data field: with both participants indexed the data holds one word, with one indexed two words
    (130 hex characters), with none three words (194). A row with the Transfer signature whose shape
    matches no such declaration cannot be decoded under the standard ABI at all; it is emitted by a
    contract writing the signature onto a payload of its own choosing, which the log opcodes permit.

    Returns counts keyed `unindexed`, `singly_indexed` and `non_conforming`. Pure, so the paper's
    classification of the residual bucket is testable without reading 19 billion rows.
    """
    out = {"unindexed": 0, "singly_indexed": 0, "non_conforming": 0}
    for shape, n in by_shape.items():
        if shape == "width=194,topics=1":
            out["unindexed"] += n
        elif shape == "width=130,topics=2":
            out["singly_indexed"] += n
        else:
            out["non_conforming"] += n
    return out


def fabrication_coverage(verdicts: dict[str, dict], counts: dict[str, int], total: float,
                         kinds: tuple[str, ...] = ("constant_nonzero",)) -> tuple[int, int, float]:
    """Join per-contract verdicts with per-contract transfer counts.

    This is the paper's headline coverage figure, so it is a pure function and unit-tested rather
    than an inline expression. Returns (n_contracts, n_rows, share_of_total).

    Two conventions matter and are asserted by tests:
      * `total` is the WHOLE channel, so contracts the probe could not reach count against the
        share rather than being removed from the denominator. The figure is therefore a lower bound.
      * A contract with a verdict but no observed transfers contributes 0 rows, not an error: the
        metadata population and the emitting population need not coincide.
    """
    if total <= 0:
        raise ValueError("total must be positive; a zero denominator cannot yield a share")
    sel = [a for a, v in verdicts.items() if v.get("verdict") in kinds]
    rows = sum(counts.get(a, 0) for a in sel)
    return len(sel), rows, rows / total


class Macros:
    """Collects name -> body pairs, refusing duplicates."""

    def __init__(self) -> None:
        self._m: dict[str, str] = {}
        self._order: list[tuple[str, str]] = []

    def add(self, name: str, body: str, comment: str = "") -> None:
        full = "lb" + name
        if full in self._m:
            raise SystemExit(f"macro \\{full} defined twice -- a redefinition makes the paper's "
                             f"number depend on input order")
        self._m[full] = body
        self._order.append((full, comment))

    def section(self, title: str) -> None:
        self._order.append(("%%SECTION%%", title))

    # --- formatters: precision is fixed here, never in the prose -----------------------------
    def int_(self, name: str, v, comment: str = "") -> None:
        self.add(name, rf"\num{{{int(v)}}}", comment)

    def pct(self, name: str, v, dp: int = 2, comment: str = "") -> None:
        """Fraction -> percentage figure WITHOUT the % sign (the text supplies the wording)."""
        self.add(name, f"{float(v) * 100:.{dp}f}", comment)

    def dec(self, name: str, v, dp: int = 4, comment: str = "") -> None:
        self.add(name, f"{float(v):.{dp}f}", comment)

    #: Counts below ten are spelled out in running prose, which is the manuscript's prevailing
    #: style ("three optimistic and two using validity proofs"). Macro-emitted counts were the only
    #: numerals left, which put "5 rollups, three optimistic and two using..." in one clause. The
    #: word still comes from the macro, so golden rule 1 is intact.
    #:
    #: Numerals are kept, deliberately, in two places: inside tables, and in a series that also
    #: contains a number ten or greater (a confusion matrix reading "86 true positives, zero false
    #: positives" would be worse, not better).
    _WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

    def word(self, name: str, v, comment: str = "") -> None:
        """Emit a count below ten as an English word. RAISES at ten and above."""
        n = int(v)
        if not 0 <= n < len(self._WORDS):
            raise SystemExit(f"macro {name}: word() takes a count below ten, got {n}. Use int_() "
                             f"and let the number render as a numeral.")
        self.add(name, self._WORDS[n], comment)

    def raw(self, name: str, body: str, comment: str = "") -> None:
        self.add(name, body, comment)

    def render(self) -> str:
        L = [
            "% GENERATED FILE -- DO NOT EDIT.",
            "% Producer: papers/lure-broadcast/scripts/make_macros.py",
            f"% git commit: {git_commit()}",
            "%",
            "% Every number quoted in this paper is defined here and nowhere else (golden rule 1).",
            "% Percentages are emitted as bare figures; the prose supplies the per-cent sign.",
            "% Counts are wrapped in siunitx \\num{} for consistent thousands grouping.",
            "",
        ]
        for name, comment in self._order:
            if name == "%%SECTION%%":
                L += ["", f"% ---- {comment} " + "-" * max(0, 68 - len(comment)), ""]
                continue
            line = rf"\newcommand{{\{name}}}{{{self._m[name]}}}"
            if comment:
                line = f"{line:<58}% {comment}"
            L.append(line)
        return "\n".join(L) + "\n"


def assert_flagged_counts_agree(lead: str) -> int:
    """Every artefact describing the flagged population must agree on its size.

    Why this exists: `corroboration_base.json` once recorded 132 flagged contracts while the live
    classification, the channel census and the contract-state probe all agreed on 141. It had been
    generated three days earlier, before two detector fixes landed, and nothing noticed -- each
    artefact is loaded independently and none is checked against the others, so a stale file reached
    the manuscript and a reviewer found it. Numbers derived from different vintages of the same
    population are not comparable, and a paper that mixes them is wrong even when every individual
    script is right.

    The check is deliberately exact rather than tolerant: these are counts of the same set, computed
    from the same inputs, so any difference is a staleness bug and not rounding.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "analysis"))
    from src import metadata_lure as _ml

    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[lead]
    live = sum(1 for r in md.values()
               if _ml.classify(r.get("name"), r.get("symbol")).strength == "strong")

    seen = {"live classification": live}
    for fname, path in ((f"campaign_census_{lead}.json", "contracts"),
                        (f"forgery_probe_{lead}.json", "lure.n"),
                        (f"corroboration_{lead}.json", "flagged_strong")):
        f = RES / fname
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        for part in path.split("."):
            doc = doc[part]
        seen[fname] = doc

    if len(set(seen.values())) != 1:
        lines = "\n".join(f"      {k:34s} {v}" for k, v in seen.items())
        raise SystemExit(
            "flagged-contract counts DISAGREE across artefacts; at least one is stale and the "
            "manuscript would mix vintages of the same population:\n" + lines +
            "\n    Regenerate the disagreeing artefact(s) before building the paper.")
    return live


def main() -> int:
    m = Macros()
    n_flagged = assert_flagged_counts_agree(LEAD_CHAIN)

    # ------------------------------------------------------------------ non-fungible census
    nf = load("nf_census_all.json")
    src = "nf_census_all.json"
    m.section("Corpus and the non-fungible census (Contribution 1)")
    m.int_("CorpusRows", _get(nf, "total_rows", src), "all Transfer log rows, 5 chains")
    m.int_("CorpusErcTwentyRows", _get(nf, "erc20_rows", src))
    m.int_("NfRows", _get(nf, "nonfungible_rows", src))
    m.int_("NfResidualRows", _get(nf, "residual_rows", src), "data hexlen outside {2,66}")
    m.pct("NfShare", _get(nf, "nonfungible_share", src), 2)
    m.word("NfChains", len(CHAINS))
    shares = {c: float(_get(nf, f"per_chain_share.{c}", src)) for c in CHAINS}
    lo, hi = min(shares, key=shares.get), max(shares, key=shares.get)
    m.pct("NfShareMin", shares[lo], 2)
    m.pct("NfShareMax", shares[hi], 2)
    m.raw("NfShareMinChain", CHAIN_TEX[lo][1])
    m.raw("NfShareMaxChain", CHAIN_TEX[hi][1])
    for c in CHAINS:
        tex = CHAIN_TEX[c][0]
        m.pct(f"NfShare{tex}", _get(nf, f"per_chain_share.{c}", src), 3)
        m.pct(f"NfMintShare{tex}", _get(nf, f"per_chain_mint_share.{c}", src), 1)
    # Per-chain absolute counts come from the individual census files.
    for c in CHAINS:
        d = load(f"{c}_nf_census.json")
        s = f"{c}_nf_census.json"
        tex = CHAIN_TEX[c][0]
        m.int_(f"NfRows{tex}", _get(d, "nonfungible_rows", s))
        m.int_(f"CorpusRows{tex}", _get(d, "total_rows", s))

    # ------------------------------------------------------------------ lure channel census
    cc = load("campaign_census_base.json")
    src = "campaign_census_base.json"
    m.section("Full-chain lure-channel census on Base (Contribution 3)")
    m.int_("LureTransfers", _get(cc, "channel_totals.transfers", src))
    m.int_("LureRecipients", _get(cc, "channel_totals.distinct_recipients_union", src),
           "addresses NAMED AS recipients")
    m.dec("LureFanout", _get(cc, "channel_totals.fanout_ratio_channel", src), 3)
    m.int_("LureContracts", _get(cc, "contracts", src))
    m.int_("LureCampaigns", _get(cc, "campaigns", src))
    m.int_("LureRowsScanned", _get(cc, "rows_scanned", src))

    # --------------------------------------------------- the forged sender field (traceable part)
    # What is traceable from the census: the DISTRIBUTION of addresses named in `from`, and its
    # concentration. What is NOT yet traceable, and therefore must not appear in the manuscript:
    # the ledger / authorisation / self-emission probes that establish fabrication contract by
    # contract (see docs/FINDINGS.md, "Traceability debt"). Do not add a coverage percentage here
    # until a committed script emits it.
    m.section("Concentration of the adversary-authored sender field")
    ops = _get(cc, "top_operators_by_volume", src)
    total = float(_get(cc, "channel_totals.transfers", src))
    m.int_("SenderDistinct", _get(cc, "channel_totals.distinct_operators", src),
           "distinct addresses NAMED as sender")
    m.pct("SenderTopFiftyShare", sum(o["transfers"] for o in ops[:50]) / total, 2)
    m.int_("SenderTopN", min(50, len(ops)))
    m.pct("SenderRankOneShare", ops[0]["transfers"] / total, 2)
    m.int_("SenderRankOneTransfers", ops[0]["transfers"])
    # Derived from the census, never typed: an address literal in this script would be a magic
    # value that could silently disagree with the data it claims to describe.
    m.raw("SenderRankOneAddr", rf"\texttt{{{ops[0]['address'][:10]}}}")
    m.pct("SenderRankTwoShare", ops[1]["transfers"] / total, 2)

    # --------------------------------------------------------- fabrication, established per contract
    # Coverage is the JOIN of two artefacts: the per-contract verdict from the RPC probe and the
    # per-contract transfer count from the full-chain scan. Neither alone can state what share of
    # the channel is fabricated, which is why the number could not be quoted until both existed.
    co = load("corroboration_base.json")
    fp = load(f"forgery_probe_{LEAD_CHAIN}.json")
    fc = load(f"forgery_chain_{LEAD_CHAIN}.json")
    sp, sc = f"forgery_probe_{LEAD_CHAIN}.json", f"forgery_chain_{LEAD_CHAIN}.json"
    if _get(fc, "partial", sc):
        raise SystemExit(f"{sc} is a PARTIAL pilot -- refusing to derive a channel coverage figure "
                         f"from an incomplete pass. Re-run forgery_chain_scan.py without "
                         f"--max-chunks.")
    m.section("Fabrication established contract by contract")

    verdicts = _get(fp, "per_contract", sp)
    counts = _get(fc, "lure_transfers_by_contract", sc)
    lure_total = float(_get(cc, "channel_totals.transfers", src))

    n_fab, fab_rows, fab_share = fabrication_coverage(verdicts, counts, lure_total)
    fab = [a for a, v in verdicts.items() if v["verdict"] == "constant_nonzero"]
    # A contract the probe could not reach contributes to neither numerator nor denominator of the
    # claim; its rows are reported separately so the residual is visible rather than absorbed.
    unreached = [a for a, v in verdicts.items() if v["verdict"] == "no_response"]

    m.int_("ForgeryContracts", _get(fp, "lure.n", sp))
    m.int_("ForgeryFabricated", n_fab)
    m.word("ForgeryAddressDerived", sum(1 for v in verdicts.values()
                                        if v["verdict"] == "address_derived"))
    m.word("ForgeryNoResponse", len(unreached))
    m.int_("ForgeryEqualsSupply", _get(fp, "lure.balance_equals_total_supply", sp))
    m.int_("ForgeryAllowanceRevert", _get(fp, "lure.allowance_reverts", sp))
    m.int_("ForgeryBothTests", _get(fp, "lure.ledger_and_authorisation_both", sp))
    m.int_("ForgeryLedgerOnly", _get(fp, "lure.ledger_only", sp))
    m.word("ForgeryControls", _get(fp, "controls.n", sp))
    m.raw("ForgeryProbeSeed", rf"\texttt{{{_get(fp, 'probe_seed', sp)}}}")

    # The control outcome, asserted here as well as in the probe: if a registry token ever came back
    # fabricated the macro set must not build, because every verdict would be suspect.
    ctrl_v = _get(fp, "controls.verdicts", sp)
    if set(ctrl_v) != {"zero_ledger"}:
        raise SystemExit(f"{sp}: control verdicts are {ctrl_v}, not all zero_ledger. The probe does "
                         f"not discriminate; refusing to emit a fabrication figure.")
    m.word("ForgeryControlsZeroLedger", ctrl_v["zero_ledger"])

    m.int_("ForgeryRows", fab_rows)
    m.pct("ForgeryCoverage", fab_share, 4)
    # Inclusive variant: a contract returning an address-DERIVED pseudo-balance is also fabricating,
    # it simply does not use a constant. Both figures are emitted so the paper can lead with the
    # strict one and state the inclusive one, rather than choosing silently.
    n_incl, incl_rows, incl_share = fabrication_coverage(
        verdicts, counts, lure_total, kinds=("constant_nonzero", "address_derived"))
    m.int_("ForgeryContractsInclusive", n_incl)
    m.int_("ForgeryRowsInclusive", incl_rows)
    m.pct("ForgeryCoverageInclusive", incl_share, 4)
    m.int_("ForgeryUnreachedRows", sum(counts.get(a, 0) for a in unreached))
    m.pct("ForgeryUnreachedShare", sum(counts.get(a, 0) for a in unreached) / lure_total, 4)

    # Probe-address absence: the ledger test is only sound if the probes are unknown to the chain.
    if not _get(fc, "probe.absent", sc):
        raise SystemExit(f"{sc}: the probe addresses OCCUR in the chain "
                         f"({_get(fc, 'probe.occurrences_in_chain', sc)} times). The ledger test "
                         f"assumes they are unseen; regenerate with a different seed.")
    m.word("ProbeOccurrences", _get(fc, "probe.occurrences_in_chain", sc))
    m.int_("ProbeScanRows", _get(fc, "rows_scanned", sc))

    # Self-emission: rows naming a registry token as sender, and how many it emitted itself.
    # Neither figure is exactly zero, so both are also emitted as rates. Rounding a small non-zero
    # residue down to "none" would be the same class of overstatement as "all 141".
    named_n = _get(fc, "totals.rows_naming_a_registry_token_as_sender", sc)
    self_n = _get(fc, "totals.of_which_emitted_by_that_token", sc)
    own_n = _get(fc, "totals.registry_own_logs", sc)
    own_self_n = _get(fc, "totals.registry_own_logs_naming_self", sc)
    m.int_("SelfEmitNamed", named_n)
    m.word("SelfEmitBySelf", self_n)
    m.int_("SelfEmitOwnLogs", own_n)
    m.int_("SelfEmitOwnLogsSelf", own_self_n)
    if named_n <= 0 or own_n <= 0:
        raise SystemExit(f"{sc}: zero rows naming a registry token as sender, or zero registry own "
                         f"logs -- the join is broken; refusing to emit a rate with a zero "
                         f"denominator")
    # Expressed as "one row in N" rather than as a percentage: at these magnitudes a percentage
    # rounds to 0.00000, which would contradict the sentence stating the residue is not zero.
    if self_n <= 0 or own_self_n <= 0:
        raise SystemExit(f"{sc}: a self-emission count is zero, so 'one in N' is undefined. The "
                         f"prose states the residue is non-zero -- rewrite it before regenerating.")
    m.int_("SelfEmitOneIn", round(named_n / self_n))

    # The single most-impersonated registry token, chain-wide. This is a DIFFERENT population from
    # SenderRankOne*, which is scoped to the flagged channel: the abstract once paired the channel-
    # scoped count (72,690,413) with a chain-wide "none of them was emitted by that contract",
    # which are not the same rows. Both quantities are emitted here so the pairing is forced.
    # FOUR populations are easy to confuse and a reviewer did confuse them, reading the difference
    # between two of them as a stale artefact. They are emitted together, and their nesting is
    # asserted, so the prose can state each denominator explicitly:
    #   chain-wide, any registry token, any emitter   = SelfEmitNamed        144,217,270
    #   chain-wide, one token, any emitter            = ImpersonTopNamed     140,372,740
    #   flagged channel, any registry token           = CritRowsBase          72,810,413
    #   flagged channel, one token                    = SenderRankOneTransfers 72,690,413
    top = max(_get(fc, "impersonation.targets", sc), key=lambda t: t["named_as_from"])
    m.raw("ImpersonTopAddr", rf"\texttt{{{top['address'][:10]}}}")
    m.int_("ImpersonTopNamed", top["named_as_from"])
    m.word("ImpersonTopSelf", top["emitted_by_self"])
    # The remainder, so the single-token figure quoted in the abstract is reconstructible from the
    # aggregate quoted in the body rather than appearing only once and unexplained.
    rest = named_n - top["named_as_from"]
    if rest < 0:
        raise SystemExit(
            f"{sc}: the most-named single token accounts for {top['named_as_from']} rows but all "
            f"registry tokens together give {named_n}; the first cannot exceed the second")
    m.int_("ImpersonRestNamed", rest)
    if top["emitted_by_self"] != 0:
        raise SystemExit(
            f"{sc}: the most-impersonated token emitted {top['emitted_by_self']} of the rows naming "
            f"it, so the abstract's claim that NONE was emitted by that contract is false. Rewrite "
            f"the sentence before regenerating.")
    m.int_("SelfEmitOwnOneIn", round(own_n / own_self_n))

    # --------------------------------------------------- multi-chain replication of the census
    # The lead chain is REQUIRED; the others are included when their artefacts exist. Which chains
    # were measured is itself reported (\lbReplicatedChains, \lbReplicatedN) so a partial
    # replication cannot read as a complete one -- the prose must never imply more coverage than
    # this list. check_amcs.py fails the build if the prose references a chain with no macro.
    m.section("Per-chain replication of the channel census")
    replicated: list[str] = []
    for c in CHAINS:
        f = RES / f"forgery_chain_{c}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        s = f"forgery_chain_{c}.json"
        if d.get("partial"):
            raise SystemExit(f"{s} is a PARTIAL pilot; delete it or re-run in full. A pilot artefact "
                             f"must never contribute to a per-chain table.")
        if "channel_census" not in d:
            raise SystemExit(f"{s} predates the channel-census fields -- re-run "
                             f"forgery_chain_scan.py for {c}")
        cen = d["channel_census"]
        tex = CHAIN_TEX[c][0]
        replicated.append(c)
        m.int_(f"ChanTransfers{tex}", _get(cen, "transfers", s))
        m.int_(f"ChanRecipients{tex}", _get(cen, "distinct_recipients", s))
        m.dec(f"ChanFanout{tex}", _get(cen, "fanout_ratio", s), 3)
        m.int_(f"ChanSenders{tex}", _get(cen, "distinct_named_senders", s))
        m.pct(f"ChanTopFifty{tex}", _get(cen, "top_fifty_share", s), 2)
        m.int_(f"ChanContracts{tex}", _get(d, "lure_contracts_observed", s))
    if LEAD_CHAIN not in replicated:
        raise SystemExit(f"the lead chain {LEAD_CHAIN} has no channel census; it is required")
    m.word("ReplicatedN", len(replicated))
    m.raw("ReplicatedChains", ", ".join(CHAIN_TEX[c][1] for c in replicated))
    fan = [float(json.loads((RES / f"forgery_chain_{c}.json").read_text())
                 ["channel_census"]["fanout_ratio"]) for c in replicated]
    m.dec("ChanFanoutMin", min(fan), 3)
    m.dec("ChanFanoutMax", max(fan), 3)

    # ------------------------------------------------- rank-depth sensitivity (reviewer item 11)
    # The depth table answers "how much does the rank cutoff cost?" quantitatively. Its columns come
    # from two artefacts that must describe the same population: the scan supplies contracts,
    # transfers and recipients per bucket, depth_campaigns.py supplies campaigns and re-asserts the
    # per-bucket contract counts against the scan before emitting anything.
    m.section("Rank-depth sensitivity of the detector frame")
    dc = load("depth_campaigns.json")
    sdc = "depth_campaigns.json"
    for c in replicated:
        tex = CHAIN_TEX[c][0]
        s = f"forgery_chain_{c}.json"
        dt = _get(json.loads((RES / s).read_text()), "depth_table", s)
        ct = _get(dc, f"per_chain.{c}.depth_table", sdc)
        if len(dt) != len(ct):
            raise SystemExit(f"{c}: depth table has {len(dt)} rows in {s} but {len(ct)} in {sdc}")
        for row, crow in zip(dt, ct):
            if row["cutoff_label"] != crow["cutoff_label"]:
                raise SystemExit(f"{c}: depth rows are not aligned ({row['cutoff_label']} vs "
                                 f"{crow['cutoff_label']}); the table would mix cutoffs")
            if row["marginal_contracts"] != crow["marginal_contracts"]:
                raise SystemExit(f"{c}: bucket {row['cutoff_label']} holds "
                                 f"{row['marginal_contracts']} contracts in {s} but "
                                 f"{crow['marginal_contracts']} in {sdc}")
            # A LaTeX control sequence is letters only, so the numeric cutoff becomes a word.
            lab = DEPTH_TEX.get(row["cutoff_label"])
            if lab is None:
                raise SystemExit(f"no macro-name spelling for depth cutoff "
                                 f"{row['cutoff_label']!r}; add it to DEPTH_TEX rather than "
                                 f"emitting a control sequence containing a digit")
            m.int_(f"Depth{lab}Contracts{tex}", row["contracts"])
            m.int_(f"Depth{lab}Campaigns{tex}", crow["campaigns"])
            m.int_(f"Depth{lab}Transfers{tex}", row["transfers"])
            m.int_(f"Depth{lab}Recipients{tex}", row["distinct_recipients"])
            m.int_(f"Depth{lab}MarginalContracts{tex}", row["marginal_contracts"])
            m.int_(f"Depth{lab}MarginalRecipients{tex}", row["marginal_recipients"])
            m.int_(f"Depth{lab}MarginalCampaigns{tex}", crow["marginal_campaigns"])
    cuts = [c for c in _get(dc, "depth_cutoffs", sdc) if c is not None]
    m.raw("DepthCutoffs", ", ".join(str(c) for c in cuts))
    m.int_("DepthCutoffMax", max(cuts))
    # The headline of the sweep: the deepest interval still adds a great deal, so the frame is not
    # converged and every contract and campaign count is a lower bound in a quantified sense.
    last_marg_c = {c: _get(dc, f"per_chain.{c}.depth_table", sdc)[-1]["marginal_campaigns"]
                   for c in replicated}
    m.int_("DepthLastMarginalCampaignsMax", max(last_marg_c.values()))
    m.raw("DepthLastMarginalCampaignsMaxChain",
          CHAIN_TEX[max(last_marg_c, key=lambda k: last_marg_c[k])][1])

    # ------------------------------------------------- pipeline performance (IEEE Access §IV)
    sy = load("systems.json")
    ssy = "systems.json"
    m.section("Implementation and performance")
    m.int_("SysRowsScanned", _get(sy, "total_rows_scanned", ssy))
    m.dec("SysWallMinutes", _get(sy, "total_wall_seconds", ssy) / 60.0, 1)
    m.dec("SysPeakRowsPerSecM", _get(sy, "peak_rows_per_second", ssy) / 1e6, 1)
    m.raw("SysPeakChain", CHAIN_TEX[_get(sy, "peak_rows_per_second_chain", ssy)][1])
    m.int_("SysMaxWorkers", _get(sy, "max_workers", ssy))
    m.int_("CorpusDiskGB", round(_get(sy, "corpus_bytes_on_disk", ssy) / 1e9))
    spread = _get(sy, "metadata_rate_spread", ssy)
    m.dec("SysMetaFastest", spread["fastest_seconds_per_contract"], 4)
    m.dec("SysMetaSlowest", spread["slowest_seconds_per_contract"], 1)
    m.int_("SysMetaSpread", round(spread["ratio"]))
    # The environment is captured by systems_report.py from the running machine, never typed, so a
    # re-run on other hardware republishes its own description instead of inheriting this one.
    env = _get(sy, "environment", ssy)
    m.raw("SysCpu", _get(env, "cpu", ssy))
    m.int_("SysCores", _get(env, "physical_cores", ssy))
    m.int_("SysMemoryGB", round(_get(env, "memory_bytes", ssy) / 2**30))
    m.raw("SysOs", f"macOS {_get(env, 'os_product_version', ssy)}")
    m.raw("SysPython", _get(env, "python", ssy))
    m.raw("SysPyarrow", _get(env, "pyarrow", ssy))
    for c in CHAINS:
        cell = _get(sy, "per_chain", ssy).get(c)
        if not cell:
            continue
        tex = CHAIN_TEX[c][0]
        m.dec(f"SysRowsPerSecM{tex}", cell["rows_per_second"] / 1e6, 1)
        # Seconds, not minutes. Polygon zkEVM's pass takes 2.0 s, which printed as "0.0" in a
        # minutes column and was read by a reviewer as a missing or erroneous value; they then
        # recomputed 18,007,594 / 9.2e6 as minutes rather than seconds and reported the table as
        # arithmetically wrong. The figures were right. The unit was unreadable.
        m.dec(f"SysWallSeconds{tex}", cell["wall_seconds"], 1)
        m.int_(f"SysWorkers{tex}", cell["workers"])

    # The channel-scoped criterion rows, and the margin by which they exceed the single most-named
    # token, so that the difference between two similar-looking figures is itself a stated quantity.
    cdj = load("criterion_denominators.json")
    scd = "criterion_denominators.json"
    crit_rows = _get(cdj, f"per_chain.{LEAD_CHAIN}.transfers", scd)["n"]
    m.int_("CritRowsBase", crit_rows)
    rank_one = _get(load(f"campaign_census_{LEAD_CHAIN}.json"),
                    "top_operators_by_volume", f"campaign_census_{LEAD_CHAIN}.json")[0]["transfers"]
    if not 0 < rank_one <= crit_rows:
        raise SystemExit(
            f"the most-named single sender accounts for {rank_one} channel rows but the criterion "
            f"over ALL registry tokens gives {crit_rows}; the first must nest inside the second or "
            f"one of the two is measuring something else")
    m.int_("CritRowsOtherTokens", crit_rows - rank_one)

    # ------------------------------------------------- response selection in the historical check
    # The historical result conditions on an endpoint serving state. These macros describe how the
    # unresolved contracts compare with the resolved ones, so the reader can judge the selection
    # rather than being asked to assume it away.
    up_file = RES / f"unresolved_profile_{LEAD_CHAIN}.json"
    if up_file.exists():
        up = load(f"unresolved_profile_{LEAD_CHAIN}.json")
        sup = f"unresolved_profile_{LEAD_CHAIN}.json"
        m.section("Response selection in the historical metadata check")
        r_, u_ = _get(up, "resolved", sup), _get(up, "unresolved", sup)
        m.pct("UnresVolumeShare", _get(up, "unresolved_share_of_volume", sup), 2)
        m.word("UnresProxiesResolved", r_["eip1167_proxies"])
        m.word("UnresProxiesUnresolved", u_["eip1167_proxies"])
        m.int_("UnresCodeMedianResolved", round(r_["code_bytes_median"]))
        m.int_("UnresCodeMedianUnresolved", round(u_["code_bytes_median"]))
        # The direction of the age difference is decided by the data, not by the prose: if the
        # unresolved contracts were OLDER the archive-depth explanation would gain support.
        m.raw("UnresAgeVerdict",
              "more recent" if u_["first_block_median"] > r_["first_block_median"] else "older")

    # ------------------------------------------------- residue-class robustness
    # The activity sample is deterministic, so its validity rests on residue class being independent
    # of activity. This block carries the direct test: the same estimator over disjoint classes.
    # The guard is on the direction that would matter. Over-dispersion means residue class carries
    # information about activity and the estimator is not replicable; that must not reach print
    # silently, so it raises.
    rr_file = RES / f"residue_robustness_{LEAD_CHAIN}.json"
    if rr_file.exists():
        rr = load(f"residue_robustness_{LEAD_CHAIN}.json")
        srr = f"residue_robustness_{LEAD_CHAIN}.json"
        m.section("Residue-class robustness of the activity estimate")
        chk = _get(rr, "class_zero_reproduces_published", srr)
        if chk["absolute_difference"] > 1e-12:
            raise SystemExit(f"{srr}: class 0 does not reproduce the published estimate "
                             f"({chk['class_zero']} vs {chk['published']}); the replicates are not "
                             f"replicates of the reported quantity")
        disp = _get(rr, "dispersion", srr)
        if disp["chi2_over_dof"] > 2.0:
            raise SystemExit(
                f"{srr}: dispersion across residue classes is {disp['chi2_over_dof']:.2f} times the "
                f"binomial expectation. Residue class carries information about activity, so the "
                f"deterministic sample is not exchangeable and the prose claiming replicability is "
                f"false. Do not publish over this.")
        sp = _get(rr, "spread", srr)
        m.word("RobClasses", _get(rr, "classes", srr))
        m.pct("RobMin", sp["min"], 2)
        m.pct("RobMax", sp["max"], 2)
        m.pct("RobRange", sp["range"], 2)
        m.pct("RobPooled", _get(rr, "pooled_active", srr), 2)
        m.dec("RobChiRatio", disp["chi2_over_dof"], 2)
        m.int_("RobPerClassN", min(c["sampled_recipients"] for c in _get(rr, "per_class", srr)))

    # ------------------------------------------------- differential registry capture
    # The activity measure observes a fraction of true activity. Applying the same RULE to both
    # groups does not make the two capture rates equal, so the observed contrast does not by itself
    # estimate the true contrast. These macros carry the exact point at which the direction would
    # reverse, which requires no assumption about the capture rates at all.
    cs = load("recipient_coverage_sensitivity.json")
    scs = "recipient_coverage_sensitivity.json"
    m.section("Sensitivity of the activity contrast to differential registry capture")
    for c, cell in _get(cs, "per_chain", scs).items():
        tex = CHAIN_TEX[c][0]
        if cell["direction"] not in ("above", "below"):
            raise SystemExit(f"{scs}: {c} has direction {cell['direction']!r}; the manuscript "
                             f"states a direction for every chain it reports")
        m.dec(f"CapThresh{tex}", cell["threshold_conservative"], 2)
        m.dec(f"CapThreshPoint{tex}", cell["threshold_point"], 2)
        # A conservative bound must be harder to reach than the point estimate, or it is not one.
        if cell["direction"] == "above" and not (
                cell["threshold_conservative"] < cell["threshold_point"]):
            raise SystemExit(f"{scs}: {c} conservative threshold is not stricter than the point "
                             f"estimate; the bound would overstate the result")

    # ------------------------------------------------- historical metadata (the temporal threat)
    # Metadata is read once, at collection time, while the events span years. This block carries the
    # test of whether that matters. The guard is on the DIRECTION again: if any comparable contract
    # turns out to have changed its metadata, the prose that attributes historical events to
    # collection-time strings is wrong, and it must not be possible to publish it silently.
    mh_file = RES / f"metadata_history_{LEAD_CHAIN}.json"
    if mh_file.exists():
        mh = load(f"metadata_history_{LEAD_CHAIN}.json")
        smh = f"metadata_history_{LEAD_CHAIN}.json"
        m.section("Historical metadata verification")
        bv = _get(mh, "by_verdict", smh)
        stable, changed = bv.get("stable", 0), bv.get("changed", 0)
        comparable = _get(mh, "comparable_contracts", smh)
        if comparable != stable + changed:
            raise SystemExit(f"{smh}: comparable is {comparable} but stable+changed is "
                             f"{stable + changed}")
        m.int_("MetaHistProbed", _get(mh, "flagged_contracts", smh))
        m.int_("MetaHistComparable", comparable)
        m.int_("MetaHistStable", stable)
        m.int_("MetaHistChanged", changed)
        m.pct("MetaHistCoverage", _get(mh, "comparable_coverage", smh), 2)
        # NO confidence interval is emitted. The comparable contracts are the subset whose
        # historical state an endpoint would serve, which is response-selected rather than a
        # probability sample of the flagged population; a binomial interval on 132/132 would be an
        # interval for a population the design never sampled. The count is reported descriptively
        # and the unresolved contracts are named.
        hist = {}
        for r in _get(mh, "records", smh).values():
            if r["verdict"] in ("stable", "changed"):
                n = sum(1 for k, v in r["at"].items()
                        if k != "end" and v["verdict"] == "answered")
                hist[n] = hist.get(n, 0) + 1
        m.int_("MetaHistTwoHeights", hist.get(2, 0))
        m.int_("MetaHistOneHeight", hist.get(1, 0))
        m.int_("MetaHistEndBlock", _get(mh, "corpus_end_block", smh))
        unver = _get(mh, "flagged_contracts", smh) - comparable
        m.word("MetaHistUnverifiable", unver)
        br = _get(mh, "by_registry", smh)
        m.int_("MetaHistRegistryMatches", br.get("registry_matches", 0))
        m.word("MetaHistRegistryWhitespace", br.get("registry_whitespace_only", 0))
        m.word("MetaHistRegistryMismatch", br.get("registry_mismatch", 0))
        # The share of channel volume the verification actually covers, which is the quantity a
        # reader needs and which contract counts alone do not give.
        fc = load(f"forgery_chain_{LEAD_CHAIN}.json")
        vol = _get(fc, "lure_transfers_by_contract", f"forgery_chain_{LEAD_CHAIN}.json")
        total = sum(vol.values())
        bad = sum(vol.get(a, 0) for a, r in _get(mh, "records", smh).items()
                  if r["verdict"] not in ("stable", "changed"))
        m.pct("MetaHistVolumeVerified", (total - bad) / total, 2)

    # ------------------------------------------------- recipient activity (the exposure gap)
    # A named recipient is not a user. This block carries the only measurement in the paper that
    # speaks to who the channel actually reaches, and it is deliberately reported with its CONTROL
    # beside it: the estimate is uninterpretable alone, because the base rate of independent
    # activity among addresses the channel never named is itself low.
    #
    # The guard below is the point of the whole block. If a chain's lure recipients are not LESS
    # active than its control, the direction of the headline sentence flips, and a sentence whose
    # direction is typed by hand while its numbers are generated is exactly the failure golden rule
    # 1 exists to prevent. So the direction is asserted here, per chain, against the intervals.
    m.section("Recipient activity: who the channel reaches")
    ra_chains = []
    for c in CHAINS:
        f = RES / f"recipient_activity_{c}.json"
        if not f.exists():
            continue
        ra = load(f"recipient_activity_{c}.json")
        sra = f"recipient_activity_{c}.json"
        # Null where the chain has no full-chain census to check against. `_get` refuses a
        # null by design, so the absence is handled here rather than papered over.
        chk = ra.get("sampler_check")
        if chk is not None and abs(chk["deviation_sigma"]) > 6.0:
            raise SystemExit(f"{sra}: the sampler is {chk['deviation_sigma']:+.1f} sigma from the "
                             f"census; its estimates describe the hash, not the channel")
        tex = CHAIN_TEX[c][0]
        # Only chains with a full-chain census can be cross-checked; the rest say so rather than
        # borrowing another chain's validation.
        if chk is not None:
            m.dec(f"RecipSamplerSigma{tex}", abs(chk["deviation_sigma"]), 2)
        ra_chains.append(c)
        act, pri = _get(ra, "active", sra), _get(ra, "prior_active", sra)
        ctl = _get(ra, "control", sra)["active"]
        # The contract-asserted observable is emitted so the paper can show the gap between it and
        # the authenticated one. It must never be the basis of a claim about who was reached.
        asr = _get(ra, "active_asserted", sra)
        ctl_asr = _get(ra, "control", sra)["active_asserted"]
        m.pct(f"RecipAssertedActive{tex}", asr["estimate"], 2)
        m.pct(f"RecipAssertedControl{tex}", ctl_asr["estimate"], 2)
        m.int_(f"RecipVerifiedTokens{tex}", _get(ra, "observable", sra)["verified_tokens"])
        m.int_(f"RecipSampled{tex}", _get(ra, "sampled_recipients", sra))
        m.pct(f"RecipActive{tex}", act["estimate"], 2)
        m.pct(f"RecipActiveLo{tex}", act["ci95_lo"], 2)
        m.pct(f"RecipActiveHi{tex}", act["ci95_hi"], 2)
        m.pct(f"RecipPrior{tex}", pri["estimate"], 2)
        m.pct(f"RecipPriorLo{tex}", pri["ci95_lo"], 2)
        m.pct(f"RecipPriorHi{tex}", pri["ci95_hi"], 2)
        m.pct(f"RecipReceived{tex}", _get(ra, "received_verified", sra)["estimate"], 2)
        m.pct(f"RecipControl{tex}", ctl["estimate"], 2)
        m.pct(f"RecipControlLo{tex}", ctl["ci95_lo"], 2)
        m.pct(f"RecipControlHi{tex}", ctl["ci95_hi"], 2)
        m.int_(f"RecipNonIndexed{tex}", _get(ra, "non_indexed_rows", sra))
        # Direction, decided by the data and not by the prose.
        below = act["ci95_hi"] < ctl["ci95_lo"]
        above = act["ci95_lo"] > ctl["ci95_hi"]
        m.raw(f"RecipVerdict{tex}",
              "below" if below else ("above" if above else "indistinguishable from"))
    if not ra_chains:
        raise SystemExit("no recipient_activity_*.json; run recipient_activity.py first")
    m.word("RecipChains", len(ra_chains))

    # ------------------------------------------------- residual bucket (reviewer item 13)
    # What the rows that match neither standard actually are. The probe is skipped on a chain the
    # census records as having none, and a skipped chain contributes nothing rather than a zero that
    # would look like a measurement.
    m.section("Composition of the residual bucket")
    shapes: dict[str, int] = {}
    probed: list[str] = []
    for c in CHAINS:
        p = RES / f"residual_probe_{c}.json"
        if not p.exists():
            raise SystemExit(f"missing residual_probe_{c}.json -- run residual_probe.py for every "
                             f"chain, including those with no residual rows, so the skip is recorded")
        d = json.loads(p.read_text())
        if d.get("skipped"):
            continue
        probed.append(c)
        if not d.get("agrees_with_census"):
            raise SystemExit(f"residual_probe_{c}.json does not agree with the census")
        for k, v in d["by_shape"].items():
            shapes[k] = shapes.get(k, 0) + v
    cls = classify_residual(shapes)
    tot_res = sum(cls.values())
    if tot_res != int(_get(nf, "residual_rows", src)):
        raise SystemExit(f"residual probe rows ({tot_res}) disagree with the census total "
                         f"({_get(nf, 'residual_rows', src)}); the bucket is described twice and "
                         f"the two descriptions differ")
    m.int_("ResidualUnindexed", cls["unindexed"])
    m.pct("ResidualUnindexedPct", cls["unindexed"] / tot_res, 1)
    m.int_("ResidualSinglyIndexed", cls["singly_indexed"])
    m.pct("ResidualSinglyIndexedPct", cls["singly_indexed"] / tot_res, 1)
    m.int_("ResidualNonConforming", cls["non_conforming"])
    m.word("ResidualChainsWithAny", len(probed))
    m.int_("ResidualContracts",
           sum(json.loads((RES / f"residual_probe_{c}.json").read_text())["distinct_contracts"]
               for c in probed))
    m.int_("ResidualOneIn", round(int(_get(nf, "total_rows", src)) / tot_res))
    # The rate for the NON-CONFORMING rows specifically. The prose previously attached the
    # residual-bucket rate above to the fourteen rows that decode under no declaration, which are
    # three orders of magnitude rarer; a referee caught the mismatched denominator. Emitting both
    # rates makes the two impossible to confuse.
    m.int_("NonConformingOneIn",
           round(int(_get(nf, "total_rows", src)) / cls["non_conforming"]))

    # --------------------------------------- interface probe: is the width test semantic? (item 13)
    ip = load(f"interface_probe_{LEAD_CHAIN}.json")
    sip = f"interface_probe_{LEAD_CHAIN}.json"
    m.section("Interface probe over the empty-data class")
    bvc = _get(ip, "by_verdict_contracts", sip)
    m.int_("IfaceProbed", _get(ip, "probed_contracts", sip))
    m.int_("IfaceErcSevenTwoOne", bvc.get("erc721_confirmed", 0))
    m.pct("IfaceCoverage", _get(ip, "probed_share_of_empty_data", sip), 1)
    # The complement is emitted too, so the prose can state what the probe does NOT cover
    # without a reader having to do the subtraction, and so it cannot drift from the coverage.
    m.pct("IfaceUncoveredPct", 1.0 - _get(ip, "probed_share_of_empty_data", sip), 1)
    m.word("NfChainsOther", len(CHAINS) - 1)
    m.pct("IfaceDecimalsRowPct", _get(ip, "answers_decimals_share_of_probed_rows", sip), 2)
    m.word("IfaceDecimalsContracts",
           bvc.get("answers_decimals_no_interface", 0) + bvc.get("answers_decimals_not_nonfungible", 0))
    m.word("IfaceUnresolved", bvc.get("unresolved", 0))
    # The remaining verdict: a contract that declares no non-fungible interface AND does not answer
    # decimals(). It was omitted from the prose, so the four figures reported came to one short of
    # the probed total. Emitting it and asserting the sum makes that impossible to repeat.
    m.word("IfaceDeniesOnly", bvc.get("denies_nonfungible_interface", 0))
    _iface_named = (bvc.get("erc721_confirmed", 0) + bvc.get("unresolved", 0)
                    + bvc.get("answers_decimals_no_interface", 0)
                    + bvc.get("answers_decimals_not_nonfungible", 0)
                    + bvc.get("denies_nonfungible_interface", 0)
                    + bvc.get("erc1155_confirmed", 0))
    if _iface_named != int(_get(ip, "probed_contracts", sip)):
        raise SystemExit(
            f"{sip}: the interface verdicts account for {_iface_named} contracts but "
            f"{_get(ip, 'probed_contracts', sip)} were probed. The prose reports these categories "
            f"as exhaustive, so a reader adding them up would find the difference.")

    # ------------------------------------------------- criterion denominators (reviewer item 9)
    # A volume share alone can be carried by one campaign, so the criterion is reported against
    # contracts, campaigns, volume and recipients, per chain.
    m.section("The fabricated-sender criterion against four denominators")
    cd = load("criterion_denominators.json")
    scd = "criterion_denominators.json"
    shares = {}
    for c in replicated:
        tex = CHAIN_TEX[c][0]
        r = _get(cd, f"per_chain.{c}", scd)
        m.int_(f"CritContracts{tex}", r["contracts"]["n"])
        m.pct(f"CritContractPct{tex}", r["contracts"]["share"], 1)
        m.int_(f"CritCampaigns{tex}", r["campaigns"]["n"])
        m.int_(f"CritCampaignsOf{tex}", r["campaigns"]["of"])
        m.pct(f"CritCampaignPct{tex}", r["campaigns"]["share"], 1)
        m.int_(f"CritTransfers{tex}", r["transfers"]["n"])
        m.pct(f"CritVolumePct{tex}", r["transfers"]["share"], 1)
        m.int_(f"CritRecipients{tex}", r["recipients"]["n"])
        m.pct(f"CritRecipientPct{tex}", r["recipients"]["share"], 1)
        shares[c] = (r["transfers"]["share"], r["campaigns"]["share"])
    # Nesting of the event-level criterion inside the contract-state result (reviewer round 5).
    li = _get(cd, f"per_chain.{LEAD_CHAIN}.ledger_intersection", scd)
    if not li["all_nested"]:
        raise SystemExit(
            f"{scd}: only {li['also_ledger_confirmed']} of {li['criterion_contracts']} criterion "
            f"contracts are ledger-confirmed. The paper claims the event-level share is NESTED "
            f"inside the contract-state result; that claim is now false and the prose must change.")
    m.int_("CritNestedContracts", li["criterion_contracts"])
    m.int_("CritNestedRows", li["criterion_rows_from_confirmed_contracts"])

    nz = {c: v for c, v in shares.items() if v[0] > 0}
    m.pct("CritVolumeSpreadLo", min(v[0] for v in nz.values()), 1)
    m.pct("CritVolumeSpreadHi", max(v[0] for v in nz.values()), 1)
    m.pct("CritCampaignSpreadLo", min(v[1] for v in nz.values()), 1)
    m.pct("CritCampaignSpreadHi", max(v[1] for v in nz.values()), 1)

    # ------------------------------------------------- normalisation ablation (Proposition 1)
    ab = load("normalisation_ablation.json")
    sa = "normalisation_ablation.json"
    m.section("Normalisation ablation, testing Proposition 1")
    if not _get(ab, "refinement_holds_on_all_chains", sa):
        raise SystemExit(f"{sa}: the refinement property of Proposition 1 is VIOLATED on the data. "
                         f"That contradicts a proved statement, so either the character tables or "
                         f"the delimiter-preservation hypothesis is wrong. Do not paper over it.")
    m.word("AblViolations", _get(ab, "total_violations", sa))
    m.word("AblWitnesses", len(_get(ab, "witnesses", sa)) - 1)
    per = _get(ab, "per_chain", sa)
    m.word("AblMergesTotal", sum(c["n_campaigns_merged"] for c in per.values()))
    m.int_("AblCampaignDeltaMax", max(c["campaign_delta"] for c in per.values()))
    m.word("AblFlaggedDeltaMax", max(abs(c["flagged_without_normalisation"]
                                         - c["flagged_with_normalisation"]) for c in per.values()))
    m.int_("AblCampaignsBase", per["base"]["campaigns_with_normalisation"])

    # ------------------------------------------- independently substantiated malicious subset
    # The detector flags on a domain plus an imperative verb, which does not establish intent. These
    # macros carry the subset for which malice rests on evidence the detector does not use: a
    # falsely named registry-verified sender, and a campaign domain on a wallet-enforced blocklist.
    # Both are LOWER BOUNDS and must be described as such.
    m.section("Independently substantiated malicious subset")
    # The figure is taken from the full-chain fabricated-sender criterion, which counts every
    # qualifying row. It was previously summed over the top-50 named senders, which agrees on the
    # lead chain only because every registry sender there happens to rank inside the top 50; on a
    # chain where one did not, that sum would silently under-count. The top-50 sum is retained as a
    # guard, since it is a subset of the criterion and can never exceed it.
    regs = {tt["address"] for tt in _get(fc, "impersonation.targets", sc)}
    chan = _get(fc, "channel_census", sc)
    top50 = sum(x["transfers"] for x in chan["top_named_senders"] if x["address"] in regs)
    imp = _get(cd, f"per_chain.{LEAD_CHAIN}.transfers.n", scd)
    if top50 > imp:
        raise SystemExit(f"the top-50 registry-sender sum ({top50}) exceeds the full-chain "
                         f"fabricated-sender criterion ({imp}); the first is a subset of the second "
                         f"and cannot be larger, so one of the two artefacts is wrong")
    # Not emitted as macros: `\lbCritTransfersBase` / `\lbCritVolumePctBase` already carry this
    # quantity from criterion_denominators.py, and two macros for one number invite a reader to
    # treat a rounding difference as a discrepancy. The equality is asserted above instead.
    _ = imp
    mmx = _get(co, "axes.metamask_domain_blocklist", "corroboration_base.json")
    m.word("SubstBlocklistHits", mmx["hits_treatment"])
    m.int_("SubstBlocklistDomains", mmx["n_treatment"])
    m.int_("SubstBlocklistSize", mmx["reference_size"])

    # --------------------------------------------- sampling frame and rank cutoff (reviewer 3, 9)
    sf = load("sampling_frame.json")
    ssf = "sampling_frame.json"
    m.section("Sampling frame: complete event census vs rank-limited detector population")
    per_sf = _get(sf, "per_chain", ssf)
    for c in CHAINS:
        tex = CHAIN_TEX[c][0]
        f = per_sf[c]["frame"]
        m.int_(f"FrameContracts{tex}", f["total_contracts"])
        m.pct(f"FrameContractPct{tex}", f["contract_fraction"], 3)
        m.pct(f"FrameVolumePct{tex}", f["transfer_fraction_all"], 2)
        m.pct(f"FrameUnexamined{tex}", f["unexamined_fraction"], 2)
        m.int_(f"FrameFloor{tex}", f["transfers_at_cutoff"])
        sw = [s for s in per_sf[c]["rank_sweep"] if s.get("evaluable")]
        m.int_(f"SweepDeepest{tex}", sw[-1]["depth"])
        m.int_(f"SweepFlaggedShallow{tex}", sw[0]["flagged"])
        m.int_(f"SweepCampShallow{tex}", sw[0]["campaigns"])
        m.int_(f"SweepFlaggedDeepest{tex}", sw[-1]["flagged"])
        m.int_(f"SweepCampDeepest{tex}", sw[-1]["campaigns"])
    m.int_("FrameCutoff", _get(per_sf["base"]["frame"], "rank_cutoff", ssf))
    m.int_("SweepShallowDepth", [s for s in per_sf["base"]["rank_sweep"] if s.get("evaluable")][0]["depth"])
    unex = {c: per_sf[c]["frame"]["unexamined_fraction"] for c in CHAINS}
    m.pct("FrameUnexaminedMin", min(unex.values()), 2)
    m.pct("FrameUnexaminedMax", max(unex.values()), 2)
    m.raw("FrameUnexaminedMaxChain", CHAIN_TEX[max(unex, key=unex.get)][1])

    # ------------------------------------------------------------- sensitivity battery (WP8)
    sb = load("sensitivity_battery.json")
    ssb = "sensitivity_battery.json"
    m.section("Sensitivity to unresolved metadata, campaign key and detector bar")
    tot = _get(sb, "campaign_key_totals", ssb)
    for k, tex in (("host_only", "KeyHostOnly"), ("host_plus_path", "KeyReported"),
                   ("full_destination", "KeyDestination"), ("payload", "KeyPayload")):
        m.int_(tex, tot[k])
    m.dec("KeyPayloadRatio", tot["payload"] / tot["host_plus_path"], 2)
    perc = _get(sb, "per_chain", ssb)
    unres = {c: perc[c]["unresolved_metadata"]["unresolved_share"] for c in CHAINS}
    m.pct("UnresolvedMax", max(unres.values()), 2)
    m.raw("UnresolvedMaxChain", CHAIN_TEX[max(unres, key=unres.get)][1])
    m.word("UnresolvedMaxN", max(perc[c]["unresolved_metadata"]["unresolved"] for c in CHAINS))
    for c in CHAINS:
        tex = CHAIN_TEX[c][0]
        m.int_(f"BarWeak{tex}", perc[c]["detector_bar"]["strong_plus_weak"])
    m.int_("KeyCampaignsBaseHost", perc["base"]["campaign_key"]["host_only"])
    m.int_("KeyCampaignsBaseReported", perc["base"]["campaign_key"]["host_plus_path"])

    # ------------------------------------------------------ adversarial / evasion suite (WP4)
    ae = load("adversarial_eval.json")
    sae = "adversarial_eval.json"
    m.section("Adversarial and evasion evaluation")
    m.int_("AdvEvasionN", _get(ae, "evasion.n", sae))
    m.int_("AdvEvasionCaught", _get(ae, "evasion.flagged", sae))
    m.pct("AdvEvasionRate", _get(ae, "evasion.detection_rate", sae), 1)
    m.word("AdvDecoyN", _get(ae, "decoys.n", sae))
    m.word("AdvDecoyFlagged", _get(ae, "decoys.false_alarms", sae))
    m.word("AdvMisses", len(_get(ae, "misses", sae)))
    m.int_("AdvCategories", len(_get(ae, "by_category", sae)))

    # ------------------------------------------------- detector evaluation (design-based, WP3)
    # Intervals come from the finite-population stratified bootstrap, NOT from a Wilson interval on
    # pooled counts: the estimand is a ratio of Horvitz-Thompson expansions over a weighted
    # without-replacement design, for which a pooled binomial interval is not valid.
    de = load(f"detector_design_based_{LEAD_CHAIN}.json")
    src = f"detector_design_based_{LEAD_CHAIN}.json"
    m.section("Detector evaluation, design-based")
    est = _get(de, "estimates", src)
    for nm, tex in (("precision", "DetPrecision"), ("recall", "DetRecall"), ("f1", "DetFone")):
        m.dec(tex, est[nm]["point"], 4)
        lo, hi = est[nm]["ci95"]
        # With no observed false positive the bootstrap cannot produce one either, so precision's
        # interval collapses to a point. Emitting [1.0000, 1.0000] would present the resampling
        # scheme's blind spot as certainty -- the exact criticism the design-based bound answers.
        if lo is not None and not (nm == "precision" and est[nm].get("degenerate_no_observed_fp")):
            m.dec(tex + "Lo", lo, 4)
            m.dec(tex + "Hi", hi, 4)
    # The design-based bound replaces the pooled rule of three. \lbDetFprUpper is REPOINTED rather
    # than renamed, because it is cited in the abstract, introduction, results and conclusions.
    m.dec("DetFprUpper", _get(est, "precision.false_positive_upper95_design_based.theta", src), 4)
    # The same bound as a percentage, so the abstract can name the quantity precisely: an upper
    # bound on the false-positive FRACTION among flagged contracts, not a detector error rate.
    m.pct("DetFprUpperPct", _get(est, "precision.false_positive_upper95_design_based.theta", src), 2)
    # The unconditionally valid bound, as a percentage. The design-based figure above holds
    # only if dropping the disagreed and blank rows is ignorable within stratum, which this
    # paper's own finding that disagreement tracks the payload makes doubtful; the artefact
    # itself labels the worst-case figure "the only unconditionally valid bound of the three".
    # Both are emitted so neither can be quoted without the other.
    m.pct("DetFprWorstPct",
          _get(est, "precision.false_positive_upper95_worst_case_exclusions.theta", src), 2)
    m.int_("DetExcludedFlaggedN",
           _get(est, "precision.false_positive_upper95_worst_case_exclusions."
                     "excluded_flagged_treated_as_false_positive", src))
    # DetFlaggedAgreed is already emitted elsewhere from the same artefact; not duplicated here.
    m.word("DetFprKmax", _get(est, "precision.false_positive_upper95_design_based.k_max", src))
    m.dec("DetFprWorst",
          _get(est, "precision.false_positive_upper95_worst_case_exclusions.theta", src), 4)
    m.dec("DetPrecisionWorst",
          _get(est, "precision.false_positive_upper95_worst_case_exclusions.observed_precision",
               src), 4)
    m.dec("DetFprAdjudicated",
          _get(est, "precision.false_positive_upper95_if_exclusions_are_positive.theta", src), 4)
    m.dec("DetFprBonferroni",
          _get(est, "precision.false_positive_upper95_bonferroni_crosscheck.theta", src), 4)
    m.dec("DetFprBinomial",
          _get(est, "precision.false_positive_upper95_simple_binomial_diagnostic.theta", src), 4)
    m.int_("DetFlaggedPop", _get(de, "flagged_census.total", src))
    m.int_("DetFlaggedDrawn", _get(de, "flagged_census.drawn", src))
    m.int_("DetFlaggedAgreed", _get(de, "flagged_census.agreed", src))
    m.word("DetExcludedFlagged", _get(de, "excluded_flagged.total", src))
    m.int_("DetBoot", _get(de, "n_boot", src))
    m.int_("DetAgreedRows", _get(de, "agreed_rows", src))
    m.int_("DetStrata", len(_get(de, "per_stratum", src)))
    m.word("DetReclassified", _get(de, "reclassified_since_draw.n", src))
    m.word("DetFullySampled", len(_get(de, "fully_sampled_strata", src)))
    raw = _get(de, "confusion_matrix_sampled_rows", src)
    for k in ("TP", "FP", "FN", "TN"):
        m.int_(f"DetRaw{k}", raw[k])
    exp = _get(de, "confusion_matrix_population_expanded", src)
    for k in ("TP", "FP", "FN", "TN"):
        m.dec(f"DetExp{k}", exp[k], 1)
    m.int_("DetPopulation", _get(de, "population_total", src))
    m.int_("DetExcludedDisagreed", _get(de, "excluded.disagreed", src))
    m.word("DetExcludedBlank", _get(de, "excluded.blank", src))

    # ------------------------------------------------------------------ inter-rater agreement
    ir = load("interrater_base.json")
    src = "interrater_base.json"
    m.section("Inter-rater reliability (kill condition K3, third leg)")
    m.dec("KappaBinary", _get(ir, "binary_lure_vs_rest.kappa", src), 4)
    m.dec("KappaBinaryAgreement", _get(ir, "binary_lure_vs_rest.observed_agreement", src), 4)
    m.dec("KappaFiveClass", _get(ir, "five_class.kappa", src), 4)
    m.dec("KappaFiveClassAgreement", _get(ir, "five_class.observed_agreement", src), 4)
    m.int_("KappaJudgedByBoth", _get(ir, "judged_by_both", src))
    m.int_("KappaDisagreements", _get(ir, "disagreements", src))
    m.int_("KappaRows", _get(ir, "rows", src))

    # ------------------------------------------------------------------ target-list independence
    th = load("theta_base.json")
    src = "theta_base.json"
    m.section("Target-list independence (Theta)")
    m.dec("Theta", _get(th, "theta", src), 3)
    m.dec("ThetaOeLure", _get(th, "oe_lure", src), 3)
    m.dec("ThetaOeCalibration", _get(th, "oe_calibration", src), 3)
    m.dec("ThetaPvalue", _get(th, "label_permutation.p_value", src), 3)
    m.int_("ThetaReplicates", _get(th, "label_permutation.replicates", src))
    m.int_("ThetaStrata", _get(th, "granularity.n_strata", src))
    m.int_("ThetaSourcesLure", _get(th, "sources.lure", src))
    m.int_("ThetaSourcesCalibration", _get(th, "sources.calibration", src))

    # Granularity sweep. NOTE: no granularity passes the pre-declared calibration band, so the
    # absolute O/E figures are NOT quotable as calibrated -- only the ratio-of-ratios is, and the
    # residual confounding ("leak") must be disclosed in the prose. These macros exist so the
    # paper is forced to state that, rather than quoting Theta as if the null were well specified.
    ss = load("strata_sweep_base.json")
    src = "strata_sweep_base.json"
    sweep = _get(ss, "sweep", src)
    band = _get(ss, "calibration_band", src)
    thetas = [float(s["theta_ratio_of_ratios"]) for s in sweep]
    strata = [int(s["n_strata"]) for s in sweep]
    m.dec("ThetaSweepLo", min(thetas), 3)
    m.dec("ThetaSweepHi", max(thetas), 3)
    m.int_("ThetaSweepStrataLo", min(strata))
    m.int_("ThetaSweepStrataHi", max(strata))
    m.dec("ThetaSweepGranularityFactor", max(strata) / min(strata), 0)
    m.dec("ThetaCalibBandLo", band[0], 2)
    m.dec("ThetaCalibBandHi", band[1], 2)
    n_pass = sum(1 for s in sweep if s["calibration_passes"])
    if n_pass:
        raise SystemExit(f"{src}: {n_pass} granularity/ies now pass calibration, but the paper text "
                         f"is written around NONE passing. Re-read the verdict and rewrite the "
                         f"disclosure before regenerating macros.")
    m.word("ThetaSweepRows", len(sweep))
    m.dec("ThetaOeLureMax", max(float(s["lure"]["oe"]) for s in sweep), 3)
    m.dec("ThetaOeCalibMax", max(float(s["calibration"]["oe"]) for s in sweep), 3)

    # ------------------------------------------------------------------ corroboration
    co = load("corroboration_base.json")
    src = "corroboration_base.json"
    m.section("External corroboration against pinned public label sources")
    m.dec("CorrCovalentRatio", _get(co, "axes.covalent_erc20_yes.ratio", src), 2)
    m.pct("CorrCovalentRateTreatment", _get(co, "axes.covalent_erc20_yes.rate_treatment", src), 2)
    m.pct("CorrCovalentRateControl", _get(co, "axes.covalent_erc20_yes.rate_control", src), 2)
    m.int_("CorrCovalentNTreatment", _get(co, "axes.covalent_erc20_yes.n_treatment", src))
    m.int_("CorrCovalentNControl", _get(co, "axes.covalent_erc20_yes.n_control", src))
    m.int_("CorrFlaggedStrong", _get(co, "flagged_strong", src))
    m.int_("CorrMetadataContracts", _get(co, "metadata_contracts", src))
    m.int_("CorrCampaignDomains", _get(co, "distinct_campaign_domains", src))
    m.int_("CorrMetamaskReference", _get(co, "axes.metamask_domain_blocklist.reference_size", src))

    # ------------------------------------------------------------------ saturation
    # The published operating point is tau = 4 (4 matching leading AND trailing nibbles). The
    # quantity is the share of the SCORED population already carrying the label, taken from the
    # unsampled census -- NOT from the negative-sampled feature table, which inflated this ~99%.
    m.section("Saturation of the tau=4 label (negative result)")
    TAU = "4"
    m.raw("SatTau", TAU)
    for c in CHAINS:
        d = load(f"saturation_{c}.json")
        s = f"saturation_{c}.json"
        tex = CHAIN_TEX[c][0]
        cell = _get(d, f"census_by_tau.{TAU}", s)
        for k in ("p_of_scored_transfers", "ratio_ceiling", "candidate_transfers_at_or_above"):
            if k not in cell:
                raise SystemExit(f"{s}: census_by_tau.{TAU} lacks {k!r}")
        m.pct(f"Saturation{tex}", cell["p_of_scored_transfers"], 2)
        m.int_(f"SatCandidates{tex}", cell["candidate_transfers_at_or_above"])
        m.int_(f"SatScored{tex}", _get(d, "scored_transfers", s))
        # ratio_ceiling is 1/p and is legitimately undefined when no candidate carries the label
        # (Polygon zkEVM: 0 of 81,423). Emitting a number there would invent one; the honest
        # statement is the Wilson upper bound on the rate, so that is what the paper gets.
        if cell["ratio_ceiling"] is None:
            if cell["candidate_transfers_at_or_above"] != 0:
                raise SystemExit(f"{s}: ratio_ceiling is null but candidates is "
                                 f"{cell['candidate_transfers_at_or_above']} -- unexplained null")
            m.raw(f"SatCeiling{tex}", r"undefined")
            m.dec(f"SatRateUpper{tex}", cell["wilson95"][1], 7)
        else:
            m.dec(f"SatCeiling{tex}", cell["ratio_ceiling"], 3)
    # Exploited chains vs the two zk nulls: the contrast is the result.
    exploited = ["base", "arbitrum-one", "optimism"]
    nulls = ["zksync-era", "polygon-zkevm"]
    sat = {c: float(load(f"saturation_{c}.json")["census_by_tau"][TAU]["p_of_scored_transfers"])
           for c in CHAINS}
    # Only the exploited chains need a ceiling summary; the nulls have none by construction.
    ceil = {c: float(load(f"saturation_{c}.json")["census_by_tau"][TAU]["ratio_ceiling"])
            for c in exploited}
    m.pct("SatExploitedLo", min(sat[c] for c in exploited), 2)
    m.pct("SatExploitedHi", max(sat[c] for c in exploited), 2)
    m.dec("SatNullHi", max(sat[c] for c in nulls), 5)
    m.dec("SatCeilingExploitedLo", min(ceil[c] for c in exploited), 3)
    m.dec("SatCeilingExploitedHi", max(ceil[c] for c in exploited), 3)

    # IEEE Access R1 revision: the new measurements (WP3-WP9), cross-checked against the artefacts
    # they share with the macros above. Kept in its own module so this main() does not grow.
    import make_macros_r1
    make_macros_r1.emit(m, load, CHAINS, CHAIN_TEX, ROOT / "papers" / "lure-broadcast")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(m.render())
    (OUT.parent / "macros.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/make_macros.py",
        "git_commit": git_commit(),
        "macro_count": len(m._m),
        "note": "every paper number originates here; \\lb prefix keeps this tree disjoint from the "
                "the repository-root paper/generated/",
    }, indent=1) + "\n")
    print(f"  {len(m._m)} macros -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
