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

## Annotation data

Rater sheets are published under pseudonymous rater IDs. The answer key is published only after
every rater's sheet was received.

## Citation and licence

Cite the paper. The archived release has DOI `[to be minted on release]`. Code and data are under the
MIT licence (`LICENSE`).
