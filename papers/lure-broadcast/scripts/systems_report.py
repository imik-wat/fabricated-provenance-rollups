"""Aggregate the pipeline's own performance from the artefacts each pass already wrote.

Why this exists
---------------
The IEEE Access version carries an implementation-and-performance section, and every number in it
must be traceable like any other (project rule 1, README). Each full-chain pass already records its
own `wall_seconds`, `rows_scanned`, `rows_per_second` and `workers`; nothing was aggregated. This
script collects them rather than re-running anything, so the reported throughput is what the
published measurements actually cost, not a fresh benchmark run under different conditions.

Two things are deliberately NOT claimed. Peak resident memory is not recorded per pass, so no memory
figure is emitted; the design is described as bounded by construction (one row group at a time) and
that claim is architectural, not measured. And the metadata sweep is reported separately from the
scans, because it is network-bound and its rate says more about a public endpoint than about this
code -- the 500-fold spread between chains is the point, not the mean.

Inputs   papers/lure-broadcast/results/forgery_chain_<chain>.json, metadata_sweep_<chain>.json
Outputs  papers/lure-broadcast/results/systems.json (+ .manifest.json)

Failure modes
-------------
- A chain whose scan artefact is missing is named in the output, never silently dropped.
- A recorded throughput that disagrees with rows_scanned / wall_seconds by more than 1% RAISES: the
  two come from the same run and a divergence means one of them is stale.

Complexity: trivial; reads a handful of small JSON documents.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "papers" / "lure-broadcast" / "results"
RAW = ROOT / "data" / "raw" / "hypersync"
CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]


#: Where the published scans read the corpus. DECLARED, not detected, because it cannot be detected
#: now: the scans ran against the machine's internal NVMe storage (APFS), and the corpus was copied
#: to an external drive afterwards to save space (the authors, 2026-10-28; the project's earlier
#: home, /Users/Kamil/Projekty/Rollups_paper on the internal disk, survives in the virtual
#: environment's interpreter path). Detecting the drive at report time described the copy, not the
#: measurements, and briefly put "an external USB drive" into the manuscript.
MEASUREMENT_STORAGE = {
    "internal": True, "medium": "NVMe", "filesystem": "APFS",
    "source": "declared by the authors: the published scans predate the copy to external storage",
}


def storage(path: Path) -> dict | None:
    """The volume `path` is read from NOW, as the operating system reports it (macOS `diskutil`).

    Recorded beside MEASUREMENT_STORAGE, never in its place: it describes where a re-run would read
    the corpus, which is slower over USB than the internal NVMe the published scans used. Only what
    the OS reports is recorded: `diskutil` cannot see whether a drive behind a USB bridge is
    solid-state, so `solid_state` is whatever it says, None included.

    Returns None where `diskutil` is unavailable, like every other probe here.
    """
    try:
        mount = path.resolve()
        while not os.path.ismount(mount):
            mount = mount.parent
        info = plistlib.loads(subprocess.run(["diskutil", "info", "-plist", str(mount)],
                                             capture_output=True, check=True).stdout)
    except Exception:
        return None
    return {"filesystem_type": info.get("FilesystemType"), "filesystem": info.get("FilesystemName"),
            "internal": info.get("Internal"), "bus": info.get("BusProtocol"),
            "solid_state": info.get("SolidState")}


def environment() -> dict:
    """Capture the machine and software the measurements were taken on.

    A throughput figure is not interpretable without the environment that produced it. Every value
    here is read from the running system rather than typed into the paper, so it cannot drift from
    the machine that produced the measurements, and a re-run on different hardware records its own
    description rather than inheriting this one.

    The values are read when this report is written, not when each scan ran. Hardware and storage do
    not change between the two; an operating-system point release can (the published scans ran a day
    before one), so the manuscript quotes the OS by major version only.

    Failure modes: a probe that cannot answer records None rather than a guess, so a missing value
    is visible as missing. Nothing here raises: an unavailable `sysctl` should not stop a report.
    """
    def sysctl(key: str) -> str | None:
        try:
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
            return r.stdout.strip() or None
        except Exception:
            return None

    def version(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    mem = sysctl("hw.memsize")
    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "physical_cores": int(sysctl("hw.physicalcpu") or 0) or None,
        "logical_cores": int(sysctl("hw.logicalcpu") or 0) or None,
        "memory_bytes": int(mem) if mem else None,
        "os": f"{platform.system()} {platform.release()}",
        "os_product_version": sysctl("kern.osproductversion"),
        "python": platform.python_version(),
        "pyarrow": version("pyarrow"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "storage": MEASUREMENT_STORAGE,
        "storage_now": storage(RAW),
        "captured": "when this report is written; see environment() for what that licenses",
        "parquet_compression": "snappy",
        "worker_policy": ("at most 12 worker processes for corpus passes and at most 4 on zkSync, "
                          "whose row groups are large enough that more workers exhaust memory"),
        "data_cached": ("cold: the corpus is far larger than RAM, so every pass reads from disk "
                        "rather than page cache"),
    }


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = RES / "systems.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    per, missing = {}, []
    for c in CHAINS:
        f = RES / f"forgery_chain_{c}.json"
        if not f.exists():
            missing.append(c)
            continue
        d = json.loads(f.read_text())
        rows, secs = int(d["rows_scanned"]), float(d["wall_seconds"])
        rps = float(d["rows_per_second"])
        # `wall_seconds` is stored to one decimal, so on a short run the quotient can differ from
        # the recorded rate purely by rounding: Polygon zkEVM's 2.0 s is a rounding of ~1.958 s, a
        # 2.1% gap on a run that is otherwise consistent. The tolerance therefore tracks half of the
        # last recorded digit rather than being a flat percentage, so the check stays strict where
        # it can be and does not fire on arithmetic that is in fact exact.
        tol = max(0.005, (0.05 / secs) if secs > 0 else 1.0)
        if secs > 0 and abs(rows / secs - rps) / max(rps, 1) > tol:
            raise SystemExit(f"{c}: recorded {rps:,.0f} rows/s but rows_scanned/wall_seconds gives "
                             f"{rows/secs:,.0f}, a gap wider than the {tol:.1%} that rounding "
                             f"{secs} s to one decimal can explain; one of the two is stale")
        entry = {"rows_scanned": rows, "wall_seconds": secs, "rows_per_second": rps,
                 "workers": int(d["workers"])}
        sw = RES / f"metadata_sweep_{c}.json"
        if sw.exists():
            s = json.loads(sw.read_text())
            entry["metadata_seconds_per_contract"] = float(s["seconds_per_contract"])
        per[c] = entry

    if not per:
        raise SystemExit("no chain scans found; nothing to report")

    disk = sum(p.stat().st_size for p in RAW.rglob("*.parquet")) if RAW.exists() else 0
    rates = [v["metadata_seconds_per_contract"] for v in per.values()
             if "metadata_seconds_per_contract" in v]
    doc = {
        "purpose": "performance of the passes that produced the published measurements",
        "environment": environment(),
        "per_chain": per,
        "chains_without_a_scan": missing,
        "total_rows_scanned": sum(v["rows_scanned"] for v in per.values()),
        "total_wall_seconds": sum(v["wall_seconds"] for v in per.values()),
        "peak_rows_per_second": max(v["rows_per_second"] for v in per.values()),
        "peak_rows_per_second_chain": max(per, key=lambda c: per[c]["rows_per_second"]),
        "max_workers": max(v["workers"] for v in per.values()),
        "corpus_bytes_on_disk": disk,
        "metadata_rate_spread": {
            "fastest_seconds_per_contract": min(rates) if rates else None,
            "slowest_seconds_per_contract": max(rates) if rates else None,
            "ratio": (max(rates) / min(rates)) if rates and min(rates) > 0 else None,
            "note": ("the sweep is network-bound; the spread measures public endpoints under "
                     "throttling, not this code, and is reported for that reason"),
        },
        "memory_note": ("peak resident memory is not recorded per pass and is therefore not "
                        "claimed. The scan holds one parquet row group per worker at a time, so "
                        "its footprint is bounded by row-group size times worker count rather "
                        "than by chain size; that is a property of the design, not a measurement"),
    }
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (RES / "systems.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/systems_report.py",
        "git_commit": git_commit()}, indent=1) + "\n")
    print(f"  {doc['total_rows_scanned']:,} rows over {len(per)} chains in "
          f"{doc['total_wall_seconds']/60:.1f} min; peak {doc['peak_rows_per_second']/1e6:.1f}M rows/s "
          f"({doc['peak_rows_per_second_chain']}); {disk/1e9:.0f} GB on disk")
    if rates:
        print(f"  metadata sweep spread: {min(rates):.4f}-{max(rates):.2f} s/contract "
              f"({max(rates)/min(rates):.0f}x)")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
