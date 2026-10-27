"""Extract ERC-20 Transfer logs from Envio HyperSync -> parquet (3xpl alternative).

Why this exists: HyperSync covers all rollups in scope INCLUDING ZKsync Era (which
3xpl lacks), is free with an API token, and streams directly to parquet. It is the
researched fallback/primary to the 3xpl dumps (see docs/DATA_SCOPE.md).

What it pulls: a BROAD scan of every ERC-20 `Transfer(address,address,uint256)` log
(topic0 = 0xddf252ad...) with NO contract allowlist, so counterfeit/fake token
contracts are included (required for fake-type poisoning detection). Columns mirror
the 3xpl parquet schema so both sources feed the same data/processed pipeline:
block_number, transaction_hash, log_index, address (token contract), topic1 (from),
topic2 (to), data (amount), + block timestamp.

Auth: ClientConfig(bearer_token=$HYPERSYNC_BEARER_TOKEN) from .env (gitignored).
Rate limit: proactive_rate_limit_sleep=True makes the client self-throttle to the
token's cap (e.g. 30 rpm) instead of erroring.

Verified endpoints (docs.envio.dev/docs/HyperSync/hypersync-supported-networks):
  arbitrum-one -> arbitrum, optimism -> optimism, base -> base,
  polygon-zkevm -> polygon-zkevm, zksync-era -> zksync.

Usage:
    python -m src.fetch_hypersync --chain zksync-era --from-block 0 --to-block 100000
    python -m src.fetch_hypersync --chain base --from-block 0            # 0 -> chain tip
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import hypersync
from hypersync import (
    BlockField,
    ClientConfig,
    FieldSelection,
    HexOutput,
    HypersyncClient,
    LogField,
    LogSelection,
    Query,
    StreamConfig,
)

TRANSFER_SIGNATURE = "Transfer(address,address,uint256)"

# Our chain keys -> HyperSync endpoint host. Keys align with docs/DATA_SCOPE.md;
# note the endpoint host differs (arbitrum-one -> arbitrum, zksync-era -> zksync).
CHAIN_ENDPOINTS = {
    "arbitrum-one": "https://arbitrum.hypersync.xyz",
    "optimism": "https://optimism.hypersync.xyz",
    "base": "https://base.hypersync.xyz",
    "polygon-zkevm": "https://polygon-zkevm.hypersync.xyz",
    "zksync-era": "https://zksync.hypersync.xyz",
    # Ethereum mainnet is NOT a rollup and is not part of the five-chain study. It is here for a
    # separate analysis. Adding it to this dict does not touch any emitter's chain list, so no
    # existing artefact moves.
    # NOTE: this endpoint returns block timestamps as raw BYTES where the rollup endpoints return
    # hex strings -- see corpus_scan.parse_hex_timestamps.
    "eth": "https://eth.hypersync.xyz",
}

# Log columns needed to reconstruct an ERC-20 transfer (from=topic1, to=topic2,
# amount=data), plus provenance (contract address, tx hash, positions).
LOG_FIELDS = [
    LogField.BLOCK_NUMBER,
    LogField.TRANSACTION_HASH,
    LogField.LOG_INDEX,
    LogField.ADDRESS,
    LogField.TOPIC0,
    LogField.TOPIC1,
    LogField.TOPIC2,
    LogField.DATA,
]
BLOCK_FIELDS = [BlockField.NUMBER, BlockField.TIMESTAMP]


def make_client(endpoint: str, token: str) -> HypersyncClient:
    return HypersyncClient(
        ClientConfig(
            url=endpoint,
            bearer_token=token,
            proactive_rate_limit_sleep=True,  # respect the token's rpm cap
        )
    )


def transfer_query(from_block: int, to_block: int | None) -> Query:
    """Broad ERC-20 Transfer scan: filter on topic0 only, no address allowlist."""
    topic0 = hypersync.signature_to_topic0(TRANSFER_SIGNATURE)
    return Query(
        from_block=from_block,
        to_block=to_block,
        logs=[LogSelection(topics=[[topic0]])],
        field_selection=FieldSelection(log=LOG_FIELDS, block=BLOCK_FIELDS),
    )


def plan_chunks(from_block: int, to_block: int, chunk_blocks: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) chunks covering [from_block, to_block). Resumable unit."""
    if chunk_blocks <= 0:
        return [(from_block, to_block)]
    chunks = []
    c = from_block
    while c < to_block:
        chunks.append((c, min(c + chunk_blocks, to_block)))
        c += chunk_blocks
    return chunks


def _chunk_dir(out_root: Path, chain: str, c0: int, c1: int) -> Path:
    return out_root / chain / f"chunk_{c0:012d}_{c1:012d}"


async def run(
    chain: str,
    from_block: int,
    to_block: int | None,
    out_root: Path,
    token: str,
    chunk_blocks: int,
) -> int:
    endpoint = CHAIN_ENDPOINTS[chain]
    client = make_client(endpoint, token)

    if to_block is None:
        to_block = await client.get_height()
        print(f"{chain}: chain tip is block {to_block:,}", flush=True)

    config = StreamConfig(hex_output=HexOutput.PREFIXED)  # 0x-prefixed hex, not raw bytes
    chunks = plan_chunks(from_block, to_block, chunk_blocks)
    print(f"{chain}: {len(chunks)} chunk(s) over [{from_block:,}..{to_block:,}]", flush=True)

    done_count = 0
    for i, (c0, c1) in enumerate(chunks, 1):
        sub = _chunk_dir(out_root, chain, c0, c1)
        marker = sub / "_DONE"
        if marker.exists():  # resume: skip completed chunks
            done_count += 1
            continue
        sub.mkdir(parents=True, exist_ok=True)
        await client.collect_parquet(str(sub), transfer_query(c0, c1), config)
        marker.write_text("ok\n")
        done_count += 1
        print(f"{chain}: chunk {i}/{len(chunks)} [{c0:,}..{c1:,}] done", flush=True)

    print(f"{chain}: complete ({done_count}/{len(chunks)} chunks) -> {out_root / chain}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch ERC-20 Transfer logs via HyperSync.")
    p.add_argument("--chain", required=True, choices=sorted(CHAIN_ENDPOINTS))
    p.add_argument("--from-block", type=int, default=0)
    p.add_argument("--to-block", type=int, default=None, help="omit -> chain tip")
    p.add_argument("--chunk-blocks", type=int, default=5_000_000,
                   help="resumable chunk size in blocks (0 = single chunk)")
    p.add_argument("--raw-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "hypersync"))
    args = p.parse_args(argv)

    token = os.environ.get("HYPERSYNC_BEARER_TOKEN")
    if not token:
        print("ERROR: HYPERSYNC_BEARER_TOKEN not set (source .env).", file=sys.stderr)
        return 2

    return asyncio.run(
        run(args.chain, args.from_block, args.to_block, Path(args.raw_dir), token, args.chunk_blocks)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
