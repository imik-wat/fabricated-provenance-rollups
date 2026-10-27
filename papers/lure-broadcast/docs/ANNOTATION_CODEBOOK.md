# Annotation codebook — metadata lure adjudication

**Version 1.0, frozen 2026-08-12.** Freeze date matters: if this document changes after annotation
begins, the κ that results is not interpretable. Amend by issuing v1.1 and re-annotating, never by
editing in place.

## What you are judging

You will see a token contract's `name()` and `symbol()` strings, exactly as they appear on-chain, plus
the chain and the contract's transfer-volume rank. **You will not see any detector output.** That is
deliberate: the whole purpose is an independent judgement to measure the detector against, and knowing
its verdict would destroy that.

The question is **not** "is this token a scam". It is narrower and more answerable:

> Do these strings exist in order to deliver a message to whoever sees them in a wallet?

A token can be a scam without being a lure (a rug pull whose name is just `PEPE`), and a lure is
identified by its *delivery function*, not by its dishonesty.

## Labels

Assign exactly one.

| Label | Meaning | Test |
|---|---|---|
| `lure` | The strings carry a destination **and** an instruction or inducement. | Would a wallet user who read this know where to go and why? |
| `lure-nodest` | An instruction or inducement, but **no** destination. | "Claim your reward" with nowhere to claim it. |
| `dest-only` | A destination with no instruction or inducement. | A bare domain that might simply be branding. |
| `legitimate` | Ordinary token metadata, including protocol-derivative names. | An LP, receipt, bridged or wrapped token; a normal project name. |
| `unclear` | You cannot decide from the strings alone. | Use this rather than guessing. |

### Destination
A URL, a bare hostname, a messaging handle (`t.me/...`, `@channel`), a shortener link, or an
obfuscated form of any of these (`example[.]com`, `example dot com`, `hxxp://`).

### Instruction or inducement
An imperative (`claim`, `visit`, `redeem`, `unlock`, `connect`, `swap`, `verify`), a reward framing
(`airdrop`, `bonus`, `giveaway`, `free`, `rewards`, `eligible`), a deadline (`within 7 days`,
`until 03.01`), or a price/amount presented as an incentive (`@$1.4537`, `$500 reward`).

## Decision rules — apply in order

1. **Read the strings only.** Do not look up the contract, the domain, or the project. The detector
   sees only these strings, so an annotator who researches externally is measuring something else.
2. **A protocol whose brand IS a domain is not a destination.** `Curve.fi 2CRV`, `aerodrome.finance`,
   `syncswap.xyz`, `balancer.fi`, `ether.fi`, `yearn.fi` → the domain is the project's name.
   Label `legitimate` unless there is also an instruction.
3. **Derivative tokens are legitimate.** Aave receipt (`aBasWETH`, `aArbWETH`) and variable-debt
   tokens, Axelar/wrapped/bridged forms (`axlUSDC`, `USDbC`), LP and position tokens
   (`SyncSwap USDC/WETH Classic LP`, `vAMM-USDC/AERO`, `Uniswap V3 Positions NFT-V1`).
   These are the known false positives of brand-registry matching and must not be labelled `lure`.
4. **Impersonation alone is not a lure.** A contract named exactly `USD Coin` / `USDC` that is not the
   real USDC is a counterfeit, but if it carries no destination and no instruction it is **not** a lure.
   Label `legitimate` — the strings are ordinary, whatever the contract's intent. (The counterfeit
   question belongs to a different measurement.)
5. **Decoration does not decide anything.** Tick marks, emoji, unusual scripts and invisible characters
   are common in both lures and ordinary tokens. Judge the content.
6. **Non-Latin scripts:** if a string uses lookalike characters (Cyrillic `с` for `c`), read it as the
   letter it resembles. `✅СIRCLE` reads as `CIRCLE`.
7. **A deadline or an amount counts as an inducement** even with no imperative verb.
8. **When genuinely torn, use `unclear`.** A forced guess adds noise to κ and to the precision estimate;
   `unclear` is informative and is reported.

## Worked examples

| Strings | Label | Why |
|---|---|---|
| `Claim OP Rewards at https://optimisam.org` / `OP` | `lure` | URL + imperative |
| `PENDLE` / `PENDLE \| t.me/s/claimpendle \| *Claim within 7 days` | `lure` | handle + imperative + deadline |
| `✅СIRCLE TOKEN DISTRIBUTION` / `t.me/s/US_POOL \| claim until 03.01` | `lure` | handle + imperative + deadline (rule 6) |
| `Optimism (OP) (@$1.4537)` / `Claim on app.optimisam.ltd` | `lure` | domain + imperative + amount |
| `Free Coin` / `FREE` | `lure-nodest` | inducement, nowhere to go |
| `www.fomox.club` / `fomox` | `dest-only` | domain, no instruction |
| `Curve.fi 2CRV` / `2CRV` | `legitimate` | rule 2 |
| `Aave Base WETH` / `aBasWETH` | `legitimate` | rule 3 |
| `USD Coin` / `USDC` (a counterfeit) | `legitimate` | rule 4 — no destination, no instruction |
| `Wrapped Ether` / `WETH` | `legitimate` | ordinary |
| `x` / `x` | `unclear` | nothing to judge |

## Protocol

- **Two annotators**, independently, no discussion until both are finished.
- Work through the file **in the order given** (it is already shuffled) and do not skip.
- **Third-pass adjudication** for every disagreement, by a third person who sees both labels.
- Report **Cohen's κ** on (a) the 5-class task and (b) the binary task `lure` vs everything else.
  The binary κ is the one the detector is measured against, since the detector is binary.
- Interpretation: κ ≥ 0.8 strong, 0.6–0.8 substantial, 0.4–0.6 moderate. **κ < 0.4 fires kill
  condition K3's third leg** — the label has no inter-rater validity and the codebook needs revision
  before any prevalence figure is published.
- Record time spent. At ~4 seconds per contract, 400 contracts is roughly 30 minutes per annotator.

## What the sample contains, and why it is not uniform

The sample is **stratified**, not random, because a uniform draw from the volume head would be
dominated by one class and give a κ that reflects the prevalence rather than the difficulty. Strata
cover the detector's three strength levels crossed with volume-rank bands, **plus an explicit negative
stratum** of contracts the detector did not flag at all. The negative stratum is what makes a
false-negative estimate possible; without it the annotation can only measure precision, never recall.

Stratum membership is recorded in the answer key, **not** in the file you annotate — otherwise it would
leak the detector's verdict (see rule 1).
