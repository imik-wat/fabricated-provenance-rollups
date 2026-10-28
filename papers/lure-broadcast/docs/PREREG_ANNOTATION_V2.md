# Pre-registration: v2 annotation study

**Registered 2026-10-27, before the sample was drawn and before any rater saw it.** The git commit
that adds this file is the registration record. Anything changed after that commit is logged under
§9, with its date and reason.

## 1. Purpose

The study answers IEEE review comments R1.1, R2.1 and R1.2, together with defect C2 (see
REVISION_PLAN_IEEE_R1.md). In the v1 study the codebook was written with the rule, both raters were
authors, the detector was revised after the labels were read, and only Base was covered. This study
measures how the frozen detector agrees with independent judgments, out-of-sample and on five chains.

## 2. Frozen detector

The verdict depends on four inputs, plus the protocol: `analysis/src/metadata_lure.py`,
`analysis/src/confusables.py`, `papers/lure-broadcast/data/tlds/tlds-alpha-by-domain.txt` and
`docs/ANNOTATION_PROTOCOL_V2.md`. Their SHA-256 digests are written into
`results/annotation_v2/sample.manifest.json` when the sample is drawn. The v2 results are computed
**only** against those digests. A later change to any input does not alter the v2 figures. If the
detector is changed, any new evaluation of it requires a new registration and a fresh sample.

## 3. Population and sample

- **Frame.** On each chain, the detector population used in the paper: the top 1,200 contracts by
  transfer count, registry-listed tokens excluded, with resolved metadata. Contracts with neither a
  name nor a symbol are excluded and counted.
- **Out-of-sample on Base.** All 372 contracts of the v1 sample are excluded. Weights describe the
  remaining population.
- **Strata.** Current detector band (strong, weak, none) × volume-rank band (0–200, 200–600,
  600–1200, 1200+), per chain.
- **Allocation.** Each stratum gets a floor of min(5, N_h). The remainder is allocated in proportion
  to √N_h and capped at N_h.
- **Targets.** Base 100, Arbitrum One 100, Optimism 100, zkSync Era 70, Polygon zkEVM 30 (400 rows; Polygon raised to 38 by deviation 1, so 408).
- **Seed:** 20261027.
- **Blinding.** Raters see only `row_id`, `name` and `symbol`. Rows from all chains are shuffled
  together.

## 4. Raters and tier

| Tier | Raters |
|---|---|
| A | Two raters who are not authors, plus a non-author adjudicator |
| B | One non-author rater, plus one author (K. Kaczyński) |
| C | Authors only |

The tier is fixed and recorded here **before any returned sheet is opened**. Each rater's
non-author status is recorded. Raters work independently under protocol v2, without the detector,
external lookups or discussion.

## 5. Fields and derived labels

Each rater gives, for every row:
- `destination` ∈ {yes, no, unclear};
- `call_to_action` ∈ {yes, no, unclear};
- `cta_language`;
- `purpose` ∈ {solicit, name, unclear}.

Two labels are derived from these:
- **Rater conjunction** = `destination = yes` ∧ `call_to_action = yes`. It is "no" when either field
  is `no`, and *unresolved* otherwise.
- **Rater purpose** = `purpose`.

A row enters the *agreed set* for a label when both raters give the same resolved value. All other
rows are excluded, counted and never imputed.

## 6. Primary analyses

Each analysis is reported per chain and pooled over chains with design weights.

- **P1. Detector `strong` against the rater conjunction.** Report the agreement-based precision
  and recall. Report the exact 95% upper bound on the number of flagged contracts that disagree,
  using `stats.stratified_exact_upper_bound` as fixed on 2026-10-27, both under ignorability and
  in the worst case (every excluded flagged row counted as a disagreement). Report the recall
  interval from the finite-population stratified bootstrap.
- **P2. Detector `strong` against rater purpose = `solicit`.** The same quantities as P1.
- **Agreement between raters.** Cohen's κ, with a 95% CI, for `destination`, `call_to_action`,
  `purpose` and the derived conjunction.

## 7. Secondary analyses (descriptive)

- **Feature-level agreement.** The detector's destination and call to action against the raters'
  answers.
- **Language gap.** Among rows that both raters mark `call_to_action = yes` in a language other
  than English, the share in which the detector found no call to action. Also reported: how many
  of those rows each rater noted as `translated` (deviation 2).
- **Interaction cases.** How raters judged the following:
  - rows whose detector call to action rests only on a lexicon word inside the host, a Telegram
    handle, or a price pattern;
  - rows whose domains the allowlist suppressed.
- **Adjudication (tier A only).** Results after adjudication are reported **separately** from
  agreed-set results, and never in place of them.

## 8. Reporting commitments

- Every figure in §6 is reported **whatever its direction**.
- The v1 recall is reported as in-sample, together with the pre-revision value.
- The detector is not modified on the basis of v2 labels before the v2 results are written into the
  paper.
- The v2 sample is not redrawn after any sheet has been returned.

## 9. Deviations

Record each one with its date, what changed and why.

**1 (2026-10-27, before the draw).** Polygon zkEVM was registered with a target of 30 rows. Its
eight non-empty strata need 38 under the registered floor of min(5, N_h), so the design was
infeasible. The script refused it before writing any output. The target was raised to **38**,
exactly the floor sum, which keeps the floor rule unchanged and adds no discretion. The total sample
becomes 408 rows. No other chain is affected: their floor sums are 48, 26, 51 and 50 against
targets of 100, 100, 100 and 70.

**2 (2026-10-27, after the draw, before any rater started).** The rater protocol moves from v2.0 to
**v2.1**.
- **What changed.** Rule 1 now allows machine translation of the name and symbol texts themselves.
  Looking up the contract, domain or project, and opening links, remain forbidden. Raters write
  `translated` in `notes` for every row where they used a translator.
- **Why.** Under v2.0, the call-to-action question already counted any language, but a rater who
  cannot read a script had no option except `unclear`. That would have undermined the
  language-gap analysis in §7.
- **What did not change.** The sample, the sheet and the answer key: no row, column or weight
  differs, because the note goes in the existing `notes` column. The draw manifest
  (`results/annotation_v2/sample.manifest.json`) records the v2.0 protocol digest the sample was
  drawn under (`e814b71e…`). Raters receive v2.1, whose SHA-256 is
  `5157f8f397007c14b3964fbc2dd3633bb6c20da1ff03f7a6e1b6a74873c911e4`. Deviation 3 superseded this
  digest before any rater started.
- **Analysis addition**, declared before any data: the count of `translated` rows (§7).

**3 (2026-10-27, before any rater started).** Raters may answer through a web page; the rater
protocol moves to **v2.2**.
- **What changed.** How raters receive and return their work. The page
  (`papers/lure-broadcast/rater_app/`, built by `scripts/build_rater_app.py`) shows the rows one at a
  time.
  - Every rule, question, answer value and example on the page is parsed from the protocol file
    itself, never retyped.
  - Every row comes from the committed `sample_blinded.csv`, in the same order. Tests compare both
    byte for byte.
  - The page saves a CSV with exactly the sheet's columns. A test confirms that Python's csv reader
    reads it identically.
  - The "translated" note becomes a tick box, written into `notes` as `translated` (or
    `translated; <note>`), the same marker the spreadsheet uses.
  - The page refuses to move to the next row until destination, call to action, language (when the
    call to action is "yes") and purpose are answered. That enforces rule 3 ("fill in every row").
    `unclear` is always available, as before.
  - The spreadsheet remains available to any rater who prefers it.
- **Why.** Fewer clerical errors: no free typing of answer values, no sorted or deleted rows, no
  spreadsheet formula risk. It is also easier for outside raters.
- **What did not change.** The rows, their order, the questions, the answer vocabulary, the
  blinding (the page carries only `row_id`, `name` and `symbol`; tests also check that no answer-key
  address appears in it) and every analysis in §6–7.
- **Independence.** Answers are kept only in the rater's own browser until the rater saves the file.
  The page has no shared storage, so no rater can see another's answers.
- **Delivery routes are recorded.** A web-page file is named `ratings_<initials>.csv` by the page.
  The route each rater used is noted when the sheet is received and reported with the results.
- Raters receive protocol v2.2, whose SHA-256 is
  `a6c79ea86c27ed2e2ffe14b3d70787d920d79cc3ad8fd13db6462fedc8e3d623`.

**4 (2026-10-27, after three further sheets arrived and before any of them was opened).** Four
sheets were returned against a design registered for two raters plus an adjudicator. Rater status,
confirmed by the corresponding author: R1, R2 and R3 are not authors; AW is an author. The study
remains tier A.
- **Primary analyses (§6) are unchanged.** They use the registered pair: R1 and the next non-author
  sheet to arrive, R2 (20:48, before R3 at 21:01 and AW at 21:02). Arrival order is the rule the first
  tier record already stated ("identified on arrival"), so no choice was made after seeing data.
- **Adjudication (§7).** R3 fills the adjudicator role. R3 rated every row blind, without seeing the
  primary raters' answers, so the adjudicated label of a row is the primary pair's agreed value where
  they agree, and R3's value otherwise. It is still reported separately and never in place of the
  agreed-set results.
- **Robustness analyses, declared now** (descriptive):
  - R1. P1, P2 and Cohen's κ for each of the three non-author pairs (R1–R2, R1–R3, R2–R3), to
    show whether any conclusion depends on which pair is primary.
  - R2. Fleiss' κ over the three non-author raters for `destination`, `call_to_action`, `purpose`
    and the derived conjunction, over rows all three answered.
  - R3. P1 and P2 on majority labels: a row enters when at least two of the three non-author raters
    give the same resolved value.
- **Author comparison, declared now** (descriptive, never in a tier A figure):
  - A1. Cohen's κ between AW and each non-author rater, and between AW and the non-author majority
    label, for the same four fields.
  - A2. Over rows with a non-author majority label, the share on which AW's conjunction matches it
    and the share on which the detector does.
- **Why.** More independent labels strengthen the evidence, and the author's sheet speaks directly to
  whether the codebook's authors read the task differently from outsiders (review R2.1). Using them
  without changing the registered primary analysis keeps the registration's guarantees.
- **What did not change.** The sample, the answer key, the detector and §6.

