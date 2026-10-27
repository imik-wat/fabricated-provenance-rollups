"""Replication record: what was extracted, when, from where, with what (IEEE review R2.14; plan WP11.1).

Reviewer 2 asked for the computational and data-processing details needed to repeat the study. The
paper already reports scan costs (systems.json). This script records the rest, read from the
extraction tree rather than typed:
  * the block range of each chain's corpus, from the chunk directory names. Chunks are half-open
    [start, end), and `end` is the chain height when the pull began, so the last block in the corpus
    is end - 1. Gaps, overlaps or a range not starting at genesis RAISE;
  * the last block carrying a transfer log and its time, from the final chunk's blocks.parquet,
    which holds only blocks with matching logs (Polygon zkEVM's stop at its sequencer shutdown);
  * when each tip was fixed, from the timestamped "pulling" line preceding the "chain tip" line in
    the pull log. A tip in the log that disagrees with the chunk names RAISES. The log is written in
    the machine's local time and is converted to UTC; on a live chain the converted time and the
    last logged block's time agree to seconds, which is the check that the conversion is right;
  * the metadata and archive endpoints, by HOSTNAME ONLY. Any URL with credentials, a query string or
    a path segment longer than a plausible route RAISES, so a keyed URL can never reach the paper;
  * the versions of the extraction client and the analysis libraries.

The HyperSync bearer token is never read. Output: papers/lure-broadcast/results/replication.json
(+ .manifest.json). Complexity: one small parquet read per chain.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = ROOT / "data" / "raw" / "hypersync"
OUT = ROOT / "papers" / "lure-broadcast" / "results" / "replication.json"
CHAINS = ("arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era")
CHUNK = re.compile(r"^chunk_(\d{12})_(\d{12})$")
PULLING = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([a-z0-9-]+)\] pulling")
TIP = re.compile(r"^([a-z0-9-]+): chain tip is block ([\d,]+)")
LIBRARIES = ("hypersync", "pyarrow", "numpy", "scipy")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def block_range(names: list[str]) -> tuple[int, int, int]:
    """(first block, end exclusive, chunk count) of contiguous half-open chunks. Raises on any gap."""
    spans = sorted((int(m.group(1)), int(m.group(2))) for m in map(CHUNK.match, names) if m)
    if not spans:
        raise ValueError("no completed chunks")
    if spans[0][0] != 0:
        raise ValueError(f"corpus starts at block {spans[0][0]}, not at genesis")
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if a1 != b0:
            raise ValueError(f"chunks [{a0}, {a1}) and [{b0}, ...) are not contiguous")
    return spans[0][0], spans[-1][1], len(spans)


def tip_fixed_at(log: str, chain: str, tip: int) -> str:
    """Timestamp of the pull that fixed `tip` for `chain`. Raises if the log never names that tip."""
    last_pull = None
    for line in log.splitlines():
        if (m := PULLING.match(line)) and m.group(2) == chain:
            last_pull = m.group(1)
        elif (m := TIP.match(line)) and m.group(1) == chain:
            if int(m.group(2).replace(",", "")) == tip:
                if last_pull is None:
                    raise ValueError(f"{chain}: tip {tip} has no timestamped pull before it")
                return last_pull
    raise ValueError(f"{chain}: the pull log never names tip {tip}; the chunks and the log disagree")


def hosts(urls: list[str]) -> list[str]:
    """Distinct hostnames, in first-seen order. Raises on anything that could carry a credential."""
    out = []
    for u in urls:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise ValueError(f"not an endpoint URL: {u[:40]}")
        if p.username or p.password or p.query or any(len(seg) > 24 for seg in p.path.split("/")):
            raise ValueError(f"an endpoint for {p.hostname} carries what may be a credential; "
                             "it must not be recorded")
        if p.hostname not in out:
            out.append(p.hostname)
    return out


def local_to_utc(stamp: str) -> str:
    """A naive local timestamp from the pull log, in UTC, under this machine's time-zone rules."""
    local = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").astimezone()
    return local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def last_block_time(chunk: Path) -> tuple[int, str]:
    """(number, ISO UTC time) of the highest block in a chunk's blocks.parquet."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(chunk / "blocks.parquet", columns=["number", "timestamp"])
    i = pc.index(t.column("number"), pc.max(t.column("number"))).as_py()
    number = int(t.column("number")[i].as_py())
    ts = int(t.column("timestamp")[i].as_py(), 16)
    return number, datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists; pass --force")
    from src import token_meta
    import forgery_probe

    log = (RAW / "_pull.log").read_text()
    chains = {}
    for c in CHAINS:
        done = [d.name for d in (RAW / c).glob("chunk_*") if (d / "_DONE").exists()]
        first, end, n = block_range(done)
        last_dir = RAW / c / f"chunk_{(max(int(CHUNK.match(x).group(1)) for x in done)):012d}_{end:012d}"
        number, when = last_block_time(last_dir)
        if number >= end:
            raise SystemExit(f"{c}: block {number} lies outside the half-open range ending at {end}")
        chains[c] = {
            "first_block": first, "end_block_exclusive": end, "last_block": end - 1, "chunks": n,
            "last_block_with_transfer_log": number, "last_block_with_transfer_log_utc": when,
            "tip_fixed_at_utc": local_to_utc(tip_fixed_at(log, c, end)),
            "metadata_hosts": hosts(token_meta.rpc_chain_for(c)),
            "archive_hosts": hosts(forgery_probe.ARCHIVE_ENDPOINTS.get(c, [])),
        }
        r = chains[c]
        print(f"  {c:14s} [{first:,}, {end:,})  last logged {number:,} at {when}  tip fixed {r['tip_fixed_at_utc']} UTC")
    result = {
        "note": "chunks are half-open [start, end); end is the chain height when the pull began, so the "
                "corpus ends at end - 1, and a probe pinned to `end` reads state one block after it",
        "extraction": {"client": "hypersync", "script": "analysis/src/fetch_hypersync.py"},
        "versions": {"python": platform.python_version(), **{lib: version(lib) for lib in LIBRARIES}},
        "chains": chains,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/replication_record.py", "git_commit": git_commit(),
        "inputs": ["data/raw/hypersync/<chain>/chunk_*", "data/raw/hypersync/_pull.log"],
    }, indent=1) + "\n")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
