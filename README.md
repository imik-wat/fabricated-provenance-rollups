# Fabricated Provenance in Token-Transfer Events: artefacts

The minimal set of code and derived data needed to reproduce the numbers in *Fabricated Provenance in
Token-Transfer Events: A Lower-Bound Measurement of a Metadata Broadcast Channel on Five Layer-Two
Rollups* (K. Kaczyński and A. Wiącek, IEEE Access).

> **Warning.** The token names and symbols in this repository are the payloads the paper measures.
> Many carry live phishing domains and instructions to claim rewards. Do not visit them.

## What is here

| Path | Contents |
|---|---|
| `papers/lure-broadcast/results/` | Every derived artefact the paper's numbers come from, each with a `.manifest.json` naming its producer |
| `papers/lure-broadcast/scripts/` | The producers of those artefacts, and the generators that turn them into the paper's numbers and tables |
| `papers/lure-broadcast/data/` | Pinned inputs: the IANA TLD list, the CoinGecko NFT control list, per-chain token metadata, and the source manifest of the third-party label snapshots |
| `papers/lure-broadcast/docs/` | The annotation codebook, the v2 annotation protocol and its pre-registration |
| `analysis/src/` | The analysis library modules the scripts import, and the HyperSync extractor |
| `analysis/config/` | Token registry, decimals, metadata and contract-code caches |
| `RELEASE_MANIFEST.json` | SHA-256 of every file |

The set is derived from the paper: the artefacts its numbers are read from, their producers, and
those producers' inputs, to a fixed point. Nothing else is included.

Not redistributed:
- the raw extraction, about 600 GB of Parquet, which the extractor rebuilds (below);
- the third-party label snapshots, whose sources and digests are in `data/labels/manifest.json`;
- one 359 MB intermediate of an analysis the paper withdraws (supplement, S2). The artefact the paper
  reads is included, and `collect_separation_inputs.py` rebuilds the intermediate from the corpus.

## Reproducing the paper's numbers

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r analysis/requirements.txt
cd papers/lure-broadcast
python scripts/make_macros.py      # every number in the paper, as LaTeX macros
python scripts/make_tables.py      # the paper's data-driven tables
```

`make_macros.py` cross-checks the artefacts against one another, and raises on any disagreement.

### From the raw corpus

The scripts that scan the chains read `data/raw/hypersync/<chain>/`. Each chain's corpus is a
half-open block range from genesis, recorded in `results/replication.json`. To rebuild it, set
`HYPERSYNC_BEARER_TOKEN` and run:

```sh
python analysis/src/fetch_hypersync.py --chain base --to-block <end_block_exclusive>
```

State probes (balances, metadata) query public RPC endpoints at the block recorded in each artefact.
`results/replication.json` names the endpoints by host. Public endpoints throttle, so a re-run can
take hours, and an endpoint may answer differently today.

### Checked reproduction

Artefacts produced before the code was under version control name a commit that does not contain
their producer. `scripts/verify_rerun.py` re-ran each with its recorded parameters and compared the
output with the committed file field by field. It ignores only wall-clock times, throughputs and
commit hashes, and lists every field it ignored. The first pass is `results/verification_rerun.json`;
the runs after the fixes below are `results/verification_rerun_after_fixes.json`. A manifest whose
artefact a run reproduced carries a `reproduced` block naming the commit that did so.

What the first pass found:
- Two state probes read an endpoint's rate-limit error, returned inside a batch, as the contract's
  answer, and the forgery probe imposed one endpoint's batch size on the others. Both are fixed
  (`scripts/rpc_answers.py`). The committed artefacts came from runs that met no throttling, and the
  fixed probes reproduce them.
- The performance record (`systems.json`) predated the last run of the scans it summarises. It was
  rebuilt, which changed the reported throughput. The published scans read the corpus from internal
  NVMe storage; it has since been copied to an external USB drive (`storage_now` in the record), so
  re-run timings are not comparable with the published ones.
- Recipient activity on Arbitrum One and Optimism gained a sampler check from a census produced
  later, and the Polygon zkEVM metadata sweep had a renamed key. These artefacts are their re-run
  outputs; every estimate is unchanged.
- `metadata_history_base.json` cannot be reproduced exactly, because which contracts a public archive
  endpoint fails to answer varies from run to run. In the committed run and in the recorded re-run,
  no contract that answered reported a different name or symbol.

To check an artefact yourself, run `verify_rerun.py --run --only <artefact>` in a linked worktree
(`git worktree add`); it refuses the main checkout, because it restores every file a run touches.
The scans also read the raw corpus and a per-token volume table for four chains, neither of which is
shipped (Base's table alone is 1.1 GB). `python scripts/token_volumes.py <chain>` rebuilds a table from
the corpus under `papers/lure-broadcast/results/`, where the scans look first. Rebuilt so, all four
tables matched the ones the published scans read, contract for contract and count for count.

## Annotation data

`results/annotation_v2/` holds the second, pre-registered annotation study:
- the blinded sample and the answer key;
- the tier record and the sheet receipts;
- the registered evaluation;
- the robustness and author comparison declared in deviation 4;
- one exploratory check, labelled as such.

The rater sheets are in `data/`. Raters who are not authors appear under pseudonyms: R1 and R2 are the
registered pair, and R3 is the tie-breaker. AW is an author, as the paper states. The registration
and every deviation from it are in `docs/PREREG_ANNOTATION_V2.md`.

```sh
cd papers/lure-broadcast
python scripts/evaluate_annotation_v2.py --sheet R1=data/ratings_R1.csv --sheet R2=data/ratings_R2.csv \
    --adjudication data/ratings_R3.csv --force
python scripts/annotation_v2_robustness.py --sheet R1=data/ratings_R1.csv --sheet R2=data/ratings_R2.csv \
    --sheet R3=data/ratings_R3.csv --sheet AW=data/ratings_AW.csv --force
```

Each run draws 5,000 bootstrap replicates and takes 15 to 30 minutes. The scripts check that
`tier.json` is committed and unmodified, so run them in a clone of this repository, not in an
unpacked archive.

## Citation and licence

Cite the paper. Each GitHub release of this repository is archived on Zenodo, which assigns it a
DOI. The release the paper cites, v1.0.0, is
[10.5281/zenodo.23722352](https://doi.org/10.5281/zenodo.23722352); the concept DOI
[10.5281/zenodo.23722351](https://doi.org/10.5281/zenodo.23722351) always resolves to the latest release.
Code and data are under the MIT licence (`LICENSE`).
