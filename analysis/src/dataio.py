"""Read HyperSync parquet chunks -> clean ERC-20 transfer frames.

Bridges the raw pull (data/raw/hypersync/<chain>/chunk_*/{logs,blocks}.parquet) to the
detection pipeline. Handles the realities found in the live data (2026-08-05):
  - The broad topic0 scan captures BOTH ERC-20 and ERC-721 Transfer logs (identical event
    signature hash). ERC-20 carries the value in `data` (32 bytes); ERC-721 has empty
    `data` (tokenId is indexed). We keep only 32-byte-`data` rows -> ERC-20 only.
    (On Polygon zkEVM ~76% of Transfer logs were ERC-721 — filtering is essential.)
  - `from`/`to` are 32-byte left-padded topics (topic1/topic2) -> last 20 bytes = address.
  - `address` (token contract) is already lowercase 0x. Block `timestamp` is a hex string.

Vectorised (pandas) so it scales; zero-value is a string compare, avoiding uint256
conversion on every row. Emits one frame per chunk (bounded memory).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from collections import defaultdict

from .heuristic import ADDRESS_HEX_LEN

ERC20_DATA_HEXLEN = 2 + 64        # "0x" + 32-byte value
ZERO_DATA = "0x" + "0" * 64       # ERC-20 value == 0
FRAME_COLUMNS = [
    "chain", "block_number", "tx_hash", "log_index",
    "token", "sender", "receiver", "data", "is_zero", "timestamp",
]


def chunk_dirs(chain_dir: str | Path) -> list[Path]:
    """Completed/started chunk directories, in block order."""
    return sorted(Path(chain_dir).glob("chunk_*"))


def _addr_from_topic(series: pd.Series) -> pd.Series:
    # last 20 bytes (40 nibbles) of the 32-byte topic, re-prefixed
    return "0x" + series.str[-ADDRESS_HEX_LEN:]


def _timestamp_to_int(series: pd.Series) -> pd.Series:
    def conv(x):
        if x is None:
            return None
        return int(x, 16) if str(x).startswith("0x") else int(x)
    return series.map(conv)


_LOG_COLUMNS = ["address", "data", "topic1", "topic2", "transaction_hash",
                "log_index", "block_number"]


def _decode_logs(logs: pd.DataFrame, chain: str, bmap: dict | None) -> pd.DataFrame:
    """Filter to ERC-20 and decode a logs frame to FRAME_COLUMNS. Shared by whole-chunk
    and batched readers."""
    logs = logs[
        (logs["data"].str.len() == ERC20_DATA_HEXLEN)
        & logs["topic1"].notna()
        & logs["topic2"].notna()
    ].copy()
    logs["sender"] = _addr_from_topic(logs["topic1"])
    logs["receiver"] = _addr_from_topic(logs["topic2"])
    logs["token"] = logs["address"].str.lower()
    logs["is_zero"] = logs["data"] == ZERO_DATA
    logs["tx_hash"] = logs["transaction_hash"]
    logs["chain"] = chain
    logs["timestamp"] = logs["block_number"].map(bmap) if bmap is not None else None
    return logs[FRAME_COLUMNS].reset_index(drop=True)


def _block_timestamp_map(chunk_dir: Path) -> dict:
    blocks = pq.read_table(chunk_dir / "blocks.parquet").to_pandas()
    return dict(zip(blocks["number"], _timestamp_to_int(blocks["timestamp"])))


def load_chunk(chunk_dir: str | Path, chain: str) -> pd.DataFrame:
    """One chunk's ERC-20 transfers as a decoded frame (FRAME_COLUMNS). Loads the whole
    chunk — use iter_transfer_batches for multi-GB chunks."""
    cd = Path(chunk_dir)
    logs = pq.read_table(cd / "logs.parquet").to_pandas()
    return _decode_logs(logs, chain, _block_timestamp_map(cd))


def iter_transfer_frames(
    chain_dir: str | Path, chain: str, require_done: bool = True
) -> Iterator[pd.DataFrame]:
    """Yield one decoded ERC-20 frame per chunk (whole-chunk; small chains only)."""
    for cd in chunk_dirs(chain_dir):
        if not (cd / "logs.parquet").exists():
            continue
        if require_done and not (cd / "_DONE").exists():
            continue
        yield load_chunk(cd, chain)


def iter_transfer_batches(
    chain_dir: str | Path, chain: str, require_done: bool = True,
    batch_size: int = 1_000_000, with_timestamp: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield decoded ERC-20 frames in row-group batches — bounded memory regardless of
    chunk size (safe on zkSync's ~15 GB chunks where load_chunk OOMs). with_timestamp
    loads the per-chunk block map (skip it when detection doesn't need timestamps)."""
    for cd in chunk_dirs(chain_dir):
        lp = cd / "logs.parquet"
        if not lp.exists() or (require_done and not (cd / "_DONE").exists()):
            continue
        bmap = _block_timestamp_map(cd) if with_timestamp else None
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(columns=_LOG_COLUMNS, batch_size=batch_size):
            yield _decode_logs(batch.to_pandas(), chain, bmap)


def token_counts(
    chain_dir: str | Path, chain: str, require_done: bool = True, batch_size: int = 1_000_000
) -> dict[str, int]:
    """ERC-20 transfer count per token contract, memory-light (streamed row-group batches,
    only the address+data columns, vectorised value_counts). Safe on multi-GB chunks where
    load_chunk would OOM. Used to find top-volume tokens for the registry."""
    counts: dict[str, int] = defaultdict(int)
    for cd in chunk_dirs(chain_dir):
        lp = cd / "logs.parquet"
        if not lp.exists() or (require_done and not (cd / "_DONE").exists()):
            continue
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(columns=["address", "data"], batch_size=batch_size):
            erc20 = pc.equal(pc.utf8_length(batch.column("data")), ERC20_DATA_HEXLEN)
            addrs = pc.utf8_lower(batch.column("address").filter(erc20))
            vc = pc.value_counts(addrs)
            for pair in vc:
                counts[pair["values"].as_py()] += pair["counts"].as_py()
    return dict(counts)


def amount_int(data_hex: str) -> int:
    """Decode a single ERC-20 value on demand (kept off the hot path)."""
    h = data_hex[2:] if data_hex.startswith("0x") else data_hex
    return int(h, 16) if h else 0
