"""Read the NON-FUNGIBLE half of the Transfer corpus that the ERC-20 pipeline discards.

Why this exists
---------------
The pull captured every log matching the `Transfer(address,address,uint256)` topic0, and that signature
is shared by ERC-20 and ERC-721. `dataio._decode_logs` keeps only 32-byte-`data` rows, so roughly two
billion non-fungible rows sit on disk unread — the largest untouched asset in the corpus and the
subject of the second paper.

This is a deliberate SIBLING of `dataio._decode_logs`, not a flag on it. Adding a mode to the existing
function would put the previous paper's ERC-20 numbers one keyword argument away from silently
changing; the previous paper is under review, so that risk is not acceptable. Nothing here mutates or
re-enters the ERC-20 path, and `test_nonfungible.py` asserts the ERC-20 counts are unchanged.

The discriminant, and why `topic3` is not needed
------------------------------------------------
An ERC-721 `Transfer` indexes `tokenId` into `topic3` and leaves `data` empty; an ERC-20 `Transfer`
puts the value in `data`. `topic3` was never selected at extraction time and is ABSENT from the
parquet schema, so the two discriminants cannot be compared on this corpus at all; the comparison
below was made against the extraction API on a SAMPLE, not over the corpus, and is recorded here as
provenance rather than as a corpus-wide result. It is deliberately not quoted in any paper, since no
script under `results/` can reproduce it (golden rule 1). On that sample `data`-empty and
`topic3`-present coincided exactly on Polygon zkEVM (106,906 rows), zkSync Era (470,118) and
Optimism (15,974,343), and differed by 2 rows out of 7,728,956 on Base.

So `len(data) == 2` is the discriminant in use, and `topic3` would buy only the token identity —
which matters for "one asset moved N times" versus "N assets moved once", not for the census. What
the width test does and does not establish is measured, and published, by
`papers/lure-broadcast/scripts/interface_probe.py`: the empty-data class is SYNTACTIC, and some
empty-data contracts answer `decimals()` while claiming no non-fungible interface.

Mints and burns ARE observable without `topic3`: a mint has the zero address as sender, a burn as
receiver. That is what makes per-contract asset counts reachable from mint EVENTS.

Two honest scoping limits, to be stated in the paper rather than buried
----------------------------------------------------------------------
1. **ERC-1155 is entirely absent.** Its `TransferSingle`/`TransferBatch` topic0 values were never
   captured, and 0 of 193 probed contracts claimed interface 0xd9b67a26. This module therefore measures
   the ERC-721 subset, and `census()` reports it as such.
2. **`data == "0x"` does not prove ERC-721.** Measured on Base, 6 of 40 sampled empty-data contracts
   answer `decimals()` and do not claim ERC-721 — row-weighted, 18.9% of the queried empty-data volume.
   So `is_nonfungible` is a *syntactic* class; confirming the interface needs an on-chain ERC-165
   probe, which lives elsewhere (it needs the network; this module stays pure).

Note on a stale figure elsewhere in the tree: `dataio.py`'s docstring states "~76% of Transfer logs
were ERC-721" on Polygon zkEVM. A full census measures 9.325%. That is recorded as defect 2 in
papers/lure-broadcast/docs/DEFECTS.md and is deliberately not corrected here.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/{logs,blocks}.parquet   (read-only)

Outputs
-------
In-memory frames (NF_FRAME_COLUMNS) and census dicts. Nothing is written to disk by this module.

Failure modes
-------------
- A `data` value that is neither 2 nor 66 hex characters is counted in an explicit `residual` bucket,
  never silently dropped, and `census()` breaks the residual down by observed length.
- Rows with a null topic1/topic2 cannot yield an address and are counted as `malformed`, not dropped
  silently.
- A chunk lacking `_DONE` is skipped under `require_done` (the dataio default) so a partial pull cannot
  masquerade as a complete census; `census()` reports the chunk counts so coverage is auditable.

Complexity: O(rows) time; O(one row group) memory for the batched readers, O(distinct contracts) for
the census aggregates. Measured throughput on this machine: 30.2M rows/s projecting `data` alone,
11.6M rows/s on the four-column projection.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .dataio import ERC20_DATA_HEXLEN, _addr_from_topic, _block_timestamp_map, chunk_dirs
from .heuristic import ZERO_ADDRESS

#: "0x" -- an ERC-721 Transfer carries no data payload; tokenId is indexed into topic3.
NONFUNGIBLE_DATA_HEXLEN = 2

#: Deliberately NOT dataio.FRAME_COLUMNS: that frame carries `data` and `is_zero`, both of which are
#: meaningless here (`data` is always "0x", and "value == 0" has no referent for a non-fungible
#: transfer). A separate column set stops a caller passing one frame where the other is expected.
NF_FRAME_COLUMNS = [
    "chain", "block_number", "tx_hash", "log_index",
    "contract", "sender", "receiver", "is_mint", "is_burn", "timestamp",
]

_NF_LOG_COLUMNS = ["address", "data", "topic1", "topic2", "transaction_hash",
                   "log_index", "block_number"]


def _decode_nonfungible_logs(logs: pd.DataFrame, chain: str, bmap: dict | None) -> pd.DataFrame:
    """Filter a logs frame to non-fungible rows and decode to NF_FRAME_COLUMNS.

    Mirrors `dataio._decode_logs` in shape and vectorisation, but selects the complementary class and
    emits mint/burn flags in place of the value fields. Rows with a null topic1/topic2 are dropped here
    (they cannot yield an address); `census` counts them separately so the loss is visible.
    """
    logs = logs[
        (logs["data"].str.len() == NONFUNGIBLE_DATA_HEXLEN)
        & logs["topic1"].notna()
        & logs["topic2"].notna()
    ].copy()
    logs["sender"] = _addr_from_topic(logs["topic1"])
    logs["receiver"] = _addr_from_topic(logs["topic2"])
    logs["contract"] = logs["address"].str.lower()
    logs["is_mint"] = logs["sender"] == ZERO_ADDRESS
    logs["is_burn"] = logs["receiver"] == ZERO_ADDRESS
    logs["tx_hash"] = logs["transaction_hash"]
    logs["chain"] = chain
    logs["timestamp"] = logs["block_number"].map(bmap) if bmap is not None else None
    return logs[NF_FRAME_COLUMNS].reset_index(drop=True)


def load_nonfungible_chunk(chunk_dir: str | Path, chain: str) -> pd.DataFrame:
    """One chunk's non-fungible transfers as a decoded frame.

    Loads the whole chunk: use `iter_nonfungible_batches` on Base or zkSync, whose chunks reach tens of
    gigabytes and where this would exhaust memory.
    """
    cd = Path(chunk_dir)
    logs = pq.read_table(cd / "logs.parquet").to_pandas()
    return _decode_nonfungible_logs(logs, chain, _block_timestamp_map(cd))


def iter_nonfungible_batches(
    chain_dir: str | Path, chain: str, require_done: bool = True,
    batch_size: int = 1_000_000, with_timestamp: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield decoded non-fungible frames in row-group batches — memory bounded by one batch.

    `with_timestamp` loads the per-chunk block map, which costs a full read of blocks.parquet; skip it
    when the analysis does not need wall-clock time.
    """
    for cd in chunk_dirs(chain_dir):
        lp = cd / "logs.parquet"
        if not lp.exists() or (require_done and not (cd / "_DONE").exists()):
            continue
        bmap = _block_timestamp_map(cd) if with_timestamp else None
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(columns=_NF_LOG_COLUMNS, batch_size=batch_size):
            yield _decode_nonfungible_logs(batch.to_pandas(), chain, bmap)


def census(
    chain_dir: str | Path, chain: str, require_done: bool = True, batch_size: int = 1_000_000
) -> dict:
    """Classify EVERY Transfer log row of a chain by token standard, and count what does not fit.

    This is the paper's contribution-1 measurement, so it is deliberately exhaustive rather than
    convenient: every row lands in exactly one of `erc20`, `nonfungible` or `residual`, the three sum
    to `total_rows`, and the residual is broken down by observed `data` length so an unexpected encoding
    is visible rather than absorbed. A census that quietly dropped rows would understate the very share
    it exists to measure.

    Returns a dict with the row classification, per-chain shares, mint/burn shares of the non-fungible
    class, distinct contract/sender/recipient counts, and the chunk coverage that produced it.

    Uses pyarrow compute over a four-column projection rather than pandas, which is what makes a
    19-billion-row corpus affordable (measured 11.6M rows/s per worker on this projection).
    """
    total = erc20 = nonfung = malformed = 0
    residual_by_len: dict[int, int] = defaultdict(int)
    mints = burns = 0
    contracts: set[str] = set()
    senders: set[str] = set()
    recipients: set[str] = set()
    chunks_seen = chunks_used = 0

    zero_topic = "0x" + "0" * 24 + ZERO_ADDRESS[2:]        # 32-byte left-padded zero address

    for cd in chunk_dirs(chain_dir):
        chunks_seen += 1
        lp = cd / "logs.parquet"
        if not lp.exists() or (require_done and not (cd / "_DONE").exists()):
            continue
        chunks_used += 1
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(columns=_NF_LOG_COLUMNS, batch_size=batch_size):
            n = batch.num_rows
            total += n
            lens = pc.utf8_length(batch.column("data"))
            is20 = pc.equal(lens, ERC20_DATA_HEXLEN)
            isnf = pc.equal(lens, NONFUNGIBLE_DATA_HEXLEN)
            n20 = pc.sum(is20).as_py() or 0
            nnf = pc.sum(isnf).as_py() or 0
            erc20 += n20
            nonfung += nnf
            # Anything that is neither: bucket by length so an odd encoding is named, not lost.
            if n20 + nnf != n:
                other = pc.filter(lens, pc.and_(pc.invert(is20), pc.invert(isnf)))
                for pair in pc.value_counts(other):
                    residual_by_len[pair["values"].as_py()] += pair["counts"].as_py()

            if nnf:
                sub = batch.filter(isnf)
                t1, t2 = sub.column("topic1"), sub.column("topic2")
                # Null topics cannot yield an address; counted, not silently dropped.
                bad = pc.sum(pc.or_(pc.is_null(t1), pc.is_null(t2))).as_py() or 0
                malformed += bad
                mints += pc.sum(pc.equal(t1, zero_topic)).as_py() or 0
                burns += pc.sum(pc.equal(t2, zero_topic)).as_py() or 0
                contracts.update(pc.utf8_lower(sub.column("address")).to_pylist())
                senders.update(x for x in t1.to_pylist() if x)
                recipients.update(x for x in t2.to_pylist() if x)

    residual = sum(residual_by_len.values())
    assert erc20 + nonfung + residual == total, (
        f"row classification does not partition the input: "
        f"{erc20} + {nonfung} + {residual} != {total}"
    )
    return {
        "chain": chain,
        "chunks_seen": chunks_seen,
        "chunks_used": chunks_used,
        "total_rows": total,
        "erc20_rows": erc20,
        "nonfungible_rows": nonfung,
        "residual_rows": residual,
        "residual_by_data_hexlen": {str(k): v for k, v in sorted(residual_by_len.items())},
        "malformed_topic_rows": malformed,
        "nonfungible_share": (nonfung / total) if total else None,
        "erc20_share": (erc20 / total) if total else None,
        "mint_rows": mints,
        "burn_rows": burns,
        "mint_share_of_nonfungible": (mints / nonfung) if nonfung else None,
        "burn_share_of_nonfungible": (burns / nonfung) if nonfung else None,
        "distinct_contracts": len(contracts),
        "distinct_senders": len(senders),
        "distinct_recipients": len(recipients),
        "erc1155_covered": False,
        "scope_note": (
            "ERC-1155 TransferSingle/TransferBatch topic0 was never captured, so this covers the "
            "ERC-721 subset only. data=='0x' is a syntactic class: an on-chain ERC-165 probe is "
            "required to confirm the interface (measured: 18.9% of Base's queried empty-data volume "
            "answers decimals() and does not claim ERC-721)."
        ),
    }


def contract_counts(
    chain_dir: str | Path, chain: str, require_done: bool = True, batch_size: int = 1_000_000
) -> dict[str, int]:
    """Non-fungible transfer count per contract — the sibling of `dataio.token_counts`.

    Memory-light: streams row groups projecting only `address` and `data`, and aggregates with
    vectorised `value_counts`. This is the sampling frame for the metadata sweep, so it must cover the
    whole chain rather than a head sample.
    """
    counts: dict[str, int] = defaultdict(int)
    for cd in chunk_dirs(chain_dir):
        lp = cd / "logs.parquet"
        if not lp.exists() or (require_done and not (cd / "_DONE").exists()):
            continue
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(columns=["address", "data"], batch_size=batch_size):
            isnf = pc.equal(pc.utf8_length(batch.column("data")), NONFUNGIBLE_DATA_HEXLEN)
            addrs = pc.utf8_lower(batch.column("address").filter(isnf))
            for pair in pc.value_counts(addrs):
                counts[pair["values"].as_py()] += pair["counts"].as_py()
    return dict(counts)


def fan_out(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (contract, sender) fan-out: transfers, distinct recipients, and the ratio.

    The fan-out ratio phi = transfers / distinct recipients is the paper's discriminant between
    BROADCAST (phi -> 1, every recipient reached once, which is what a delivery channel looks like) and
    circulation or target selection (phi >> 1). Measured on Optimism's fungible lure set, phi was
    7,940,986 / 7,940,985 = 1.0000001, i.e. pure broadcast.

    Takes a decoded NF frame (NF_FRAME_COLUMNS) and returns one row per (contract, sender) with
    columns: contract, sender, transfers, recipients, phi, first_block, last_block, mints, burns.
    Raises on a frame that is missing the expected columns, rather than producing a silently empty
    result.
    """
    missing = [c for c in ("contract", "sender", "receiver", "block_number") if c not in frame.columns]
    if missing:
        raise KeyError(f"fan_out needs a non-fungible frame; missing columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=["contract", "sender", "transfers", "recipients", "phi",
                                     "first_block", "last_block", "mints", "burns"])
    g = frame.groupby(["contract", "sender"], sort=False)
    out = g.agg(
        transfers=("receiver", "size"),
        recipients=("receiver", "nunique"),
        first_block=("block_number", "min"),
        last_block=("block_number", "max"),
        mints=("is_mint", "sum"),
        burns=("is_burn", "sum"),
    ).reset_index()
    out["phi"] = out["transfers"] / out["recipients"]
    return out
