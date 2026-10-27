"""One full-chain pass supplying the four log-side facts the forgery argument needs.

What this measures, and why each is needed
------------------------------------------
1. PER-CONTRACT ROW COUNTS for the lure contracts. The probe in forgery_probe.py returns a verdict
   per contract; turning that into a statement about the channel requires weighting each verdict by
   the number of transfers that contract actually emitted. `campaign_census.py` aggregates by
   campaign, not by contract, so those weights do not exist yet.

2. PROBE-ADDRESS ABSENCE. forgery_probe.py asks the contracts for the balance of two randomly
   generated addresses. The ledger argument is only sound if those addresses are genuinely unknown to
   the chain, so their absence is MEASURED here across every row rather than asserted from the
   improbability of a collision.

3. SELF-EMISSION. For each target token, how many rows name it as sender, and how many of those were
   emitted BY it. A forged sender field predicts the second number is zero: the rows naming the token
   come from other contracts entirely. This needs no RPC and no assumption about balances.

4. SELF-HOLDING. Among the target's OWN logs, how often it appears as sender or recipient. This
   closes the "hot wallet" reading -- if the token contract were a distributor of its own supply it
   would appear in its own transfer logs.

Facts 3 and 4 together are decisive without leaving the log: the rows exist, they name the token, and
the token neither emitted them nor ever appears in its own ledger movements.

Parallelism
-----------
Row groups are the unit of work, not chunks. Measured on this corpus, Base's 10,449 row groups hold
1.048 M rows each to within 0.5%, so a row-group split is inherently balanced, whereas a chunk split
is capped by the largest chunk -- one Base chunk alone holds 18.4% of the rows, which would limit the
speed-up to about 5x however many workers were used. Every accumulator here is a sum or a counter,
so partial results merge by addition and the parallel result is identical to the serial one.

Worker count is capped per the project guardrail (<=12 for experiments, <=4 on zkSync).

Vectorisation
-------------
An earlier serial version looped in Python over matched rows, which for the registry tokens' own
logs is hundreds of millions of iterations and dominated the runtime. All counting is now vectorised:
addresses and topics are compared on their low 20 bytes via a negative-index slice, so a topic is
matched against a bare 40-character address set with a set-membership test, and "did this contract
name itself?" becomes one column equality. Nothing is decoded row by row, and there is no dependence
on the topic column's padding width.

Matching on the low 20 bytes ignores a topic's upper 12 bytes, which are zero for any correctly
padded address. A topic whose low bytes matched a target while carrying non-zero padding would still
be an assertion about that address, so this is if anything the more faithful comparison.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet   (READ-ONLY, every row group)
analysis/config/token_metadata.json
analysis/src/registry.py

Outputs
-------
papers/lure-broadcast/results/forgery_chain_<chain>.json (+ .manifest.json)

Failure modes
-------------
- Zero lure rows matched on a FULL pass -> raises (the 0x-prefix join trap). A pilot may legitimately
  match none, since the campaigns are recent and the first chunks are the earliest blocks.
- A worker raising propagates and aborts the run; a partial merge is never written.
- --max-row-groups produces a *_PARTIAL.json so a pilot cannot reach make_macros.

Complexity: one pass over every row, three columns projected, a handful of vectorised set tests per
row group, distributed across `--workers` processes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, registry  # noqa: E402

from forgery_probe import PROBE_SEED, probe_addresses  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

#: Rank cutoffs for the depth-sensitivity table. A flagged contract is assigned to the SHALLOWEST
#: bucket its volume rank falls in, so the buckets partition the population and the cumulative
#: distinct-recipient count at depth d is the union of buckets up to d. Accumulating nested unions
#: instead would multiply the memory by the number of cutoffs for no gain.
DEPTH_CUTOFFS = (100, 300, 600, 10**9)

#: Project guardrail: <=12 workers for experiments, <=4 on zkSync (machine limits, README).
MAX_WORKERS = 12
ZK_MAX_WORKERS = 4
COLUMNS = ["address", "topic1", "topic2"]

# Per-worker state, built once by the initialiser rather than pickled per task.
_G: dict = {}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def bare(addr: str) -> str:
    """Low-20-byte form: lowercase, no 0x. The join key for addresses and topics alike."""
    return addr.lower().removeprefix("0x").rjust(40, "0")


def depth_buckets(chain: str, flagged: list[str], registry: list[str]) -> dict[str, int]:
    """Map each flagged contract to the shallowest rank bucket of DEPTH_CUTOFFS it falls in.

    Ranks come from the same volume table the detector population was drawn from, with registry
    tokens excluded, so a contract's bucket is the depth at which it would first have been examined.
    Exposed at module level so the depth table's campaign counts can be derived from the identical
    assignment rather than a second, possibly divergent, reimplementation.

    Returns a mapping from bare address to bucket index. A flagged contract absent from the volume
    table is OMITTED, since it has no rank and inventing one would place it in a bucket it never
    belonged to; callers that need the population total must not infer it from this mapping.
    """
    import csv as _csv
    reg_bare = {bare(a) for a in registry}
    vt = None
    for cand in (OUT_ROOT / f"{chain}_token_volumes.csv",
                 ROOT / "results" / f"{chain}_token_volumes.csv"):
        if cand.exists():
            vt = cand
            break
    if vt is None:
        raise SystemExit(f"no volume table for {chain}; depth buckets cannot be assigned")
    ranked = [a for a, _ in sorted(((r["token"].lower(), int(r["count"]))
                                    for r in _csv.DictReader(vt.open())), key=lambda kv: -kv[1])
              if bare(a) not in reg_bare]
    rank_of = {bare(a): i + 1 for i, a in enumerate(ranked)}
    out: dict[str, int] = {}
    for a in flagged:
        rk = rank_of.get(bare(a))
        if rk is None:
            continue
        for i, cut in enumerate(DEPTH_CUTOFFS):
            if rk <= cut:
                out[bare(a)] = i
                break
    return out


def _init(lure: list[str], targets: list[str], probes: list[str], batch_size: int,
          bucket_of: dict | None = None) -> None:
    _G["lure"] = pa.array(sorted({bare(a) for a in lure}))
    _G["targets"] = pa.array(sorted({bare(a) for a in targets}))
    _G["probes"] = pa.array(sorted({bare(a) for a in probes}))
    _G["batch_size"] = batch_size
    _G["bucket_of"] = bucket_of or {}


def _scan_row_group(task: tuple[str, int]) -> dict:
    """Accumulate one row group. Pure with respect to the parent: returns partials, mutates nothing."""
    path, rg = task
    lure, targets, probes = _G["lure"], _G["targets"], _G["probes"]
    out = {
        "rows": 0, "matched": 0, "probe_hits": 0,
        "per_contract": Counter(), "named_as_from": Counter(),
        "named_as_from_self": Counter(), "own_logs": Counter(),
        "own_logs_self_named": Counter(),
        # Channel census, same pass: distinct recipients give the fan-out ratio, and the sender
        # tally gives the concentration of the adversary-authored field. Recipients are uniqued
        # WITHIN the task so inter-process traffic stays proportional to distinct values, not rows.
        "lure_recipients": np.empty(0, dtype="S40"),
        "lure_senders": Counter(),
        # recipients split by the depth bucket of the emitting contract, for the rank-sensitivity
        # table; keys are bucket indices into DEPTH_CUTOFFS
        "recips_by_bucket": {i: np.empty(0, dtype="S40") for i in range(len(DEPTH_CUTOFFS))},
        "rows_by_bucket": Counter(),
        # Fabricated-sender criterion (reviewer item 9): a broadcast row whose NAMED SENDER is a
        # registry-verified token. Attributed to the emitting contract and to the recipient, so the
        # criterion can be reported against contracts, campaigns, volume and recipients rather than
        # against volume alone, where one campaign can dominate.
        "crit_rows": 0,
        "crit_contracts": Counter(),
        "crit_recipients": np.empty(0, dtype="S40"),
    }
    tbl = pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS)
    for b in tbl.to_batches(max_chunksize=_G["batch_size"]):
        out["rows"] += b.num_rows
        # Compare on the low 20 bytes; this is width-independent and needs no padding arithmetic.
        a40 = pc.utf8_slice_codeunits(pc.utf8_lower(b.column("address")), -40)
        t1 = pc.utf8_slice_codeunits(pc.utf8_lower(b.column("topic1")), -40)
        t2 = pc.utf8_slice_codeunits(pc.utf8_lower(b.column("topic2")), -40)

        # 1. per-lure-contract counts
        m = pc.is_in(a40, value_set=lure)
        if pc.any(m).as_py():
            sub = a40.filter(m)
            out["matched"] += len(sub)
            for st in pc.value_counts(sub):
                out["per_contract"][st["values"].as_py()] += st["counts"].as_py()
            # Recipients of lure transfers: distinct count -> fan-out. The zero address is a mint
            # and is not a recipient, so it is excluded here rather than in the parent.
            r = t2.filter(m).drop_null()
            r = r.filter(pc.not_equal(r, "0" * 40))
            if len(r):
                parts = [out["lure_recipients"], np.asarray(r.to_pylist(), dtype="S40")]
                out["lure_recipients"] = np.unique(np.concatenate(parts))
            # per-bucket split: filter the matched rows by the emitting contract's bucket
            bmap = _G["bucket_of"]
            if bmap:
                sub_a = a40.filter(m).to_pylist()
                sub_r = t2.filter(m).to_pylist()
                per: dict[int, list] = {}
                for ad, rc in zip(sub_a, sub_r):
                    bi = bmap.get(ad)
                    if bi is None:
                        continue
                    out["rows_by_bucket"][bi] += 1
                    if rc and rc != "0" * 40:
                        per.setdefault(bi, []).append(rc)
                for bi, vals in per.items():
                    cur = out["recips_by_bucket"][bi]
                    out["recips_by_bucket"][bi] = np.unique(
                        np.concatenate([cur, np.asarray(vals, dtype="S40")]))
            # Fabricated-sender criterion: named sender is a registry-verified token.
            crit = pc.and_(m, pc.is_in(t1, value_set=targets))
            if pc.any(crit).as_py():
                out["crit_rows"] += pc.sum(crit).as_py()
                for st in pc.value_counts(a40.filter(crit)):
                    out["crit_contracts"][st["values"].as_py()] += st["counts"].as_py()
                cr = t2.filter(crit).drop_null()
                cr = cr.filter(pc.not_equal(cr, "0" * 40))
                if len(cr):
                    out["crit_recipients"] = np.unique(np.concatenate(
                        [out["crit_recipients"], np.asarray(cr.to_pylist(), dtype="S40")]))

            # Named senders EXCLUDE the zero address. A mint carries the zero address in the
            # sender field as a convention marker, not as an impersonation, and the concentration
            # figure this feeds is a statement about impersonation. Including it also put a
            # 129-transfer marker inside the top 50, displacing a real contract. Recipients are
            # filtered the same way, so the two sides of the census stay consistent.
            s = t1.filter(m).drop_null()
            s = s.filter(pc.not_equal(s, "0" * 40))
            if len(s):
                for st in pc.value_counts(s):
                    out["lure_senders"][st["values"].as_py()] += st["counts"].as_py()

        # 2. probe-address absence, in either participant field
        for col in (t1, t2):
            n = pc.sum(pc.is_in(col, value_set=probes)).as_py()
            if n:
                out["probe_hits"] += n

        # 3. target named as sender, and whether the target itself emitted the row
        mf = pc.is_in(t1, value_set=targets)
        if pc.any(mf).as_py():
            f_t1, f_a40 = t1.filter(mf), a40.filter(mf)
            for st in pc.value_counts(f_t1):
                out["named_as_from"][st["values"].as_py()] += st["counts"].as_py()
            self_m = pc.equal(f_t1, f_a40)
            if pc.any(self_m).as_py():
                for st in pc.value_counts(f_t1.filter(self_m)):
                    out["named_as_from_self"][st["values"].as_py()] += st["counts"].as_py()

        # 4. the target's own logs, and whether it names itself in them
        me = pc.is_in(a40, value_set=targets)
        if pc.any(me).as_py():
            o_a40, o_t1, o_t2 = a40.filter(me), t1.filter(me), t2.filter(me)
            for st in pc.value_counts(o_a40):
                out["own_logs"][st["values"].as_py()] += st["counts"].as_py()
            named = pc.or_(pc.equal(o_t1, o_a40), pc.equal(o_t2, o_a40))
            named = pc.fill_null(named, False)
            if pc.any(named).as_py():
                for st in pc.value_counts(o_a40.filter(named)):
                    out["own_logs_self_named"][st["values"].as_py()] += st["counts"].as_py()
    return out


def merge(dst: dict, src: dict) -> None:
    """Commutative merge of partials. Asserted order-independent by unit test, because parallel
    results arrive in arbitrary order and a non-commutative merge would make the totals depend on
    worker scheduling."""
    for k, v in src.items():
        if isinstance(v, Counter):
            dst[k].update(v)
        elif isinstance(v, dict) and k == "recips_by_bucket":
            for bi, arr in v.items():
                if len(arr):
                    dst[k].setdefault(bi, []).append(arr)
        elif isinstance(v, np.ndarray):
            # Append only. Uniquing here would re-sort the whole accumulated array on every one of
            # thousands of merges -- O(n log n) with n growing into the tens of millions, which
            # degrades the run instead of speeding it up. The union is taken once, at the end.
            if len(v):
                dst[k].append(v)
        else:
            dst[k] += v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cores-2, capped)")
    ap.add_argument("--max-row-groups", type=int, default=0,
                    help="pilot on the first N row groups; writes a *_PARTIAL.json")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    partial = args.max_row_groups > 0
    name = f"forgery_chain_{args.chain}" + ("_PARTIAL" if partial else "")
    out = OUT_ROOT / f"{name}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cap = ZK_MAX_WORKERS if args.chain == "zksync-era" else MAX_WORKERS
    workers = args.workers or max(1, min(cap, (os.cpu_count() or 4) - 2))
    if workers > cap:
        raise SystemExit(f"--workers {workers} exceeds the guardrail of {cap} for {args.chain}")

    md = json.loads((ROOT / "analysis" / "config" /
                     "token_metadata.json").read_text())[args.chain]
    lure = sorted({a.lower() for a, m in md.items()
                   if ml.classify(m.get("name"), m.get("symbol")).strength == "strong"})
    if not lure:
        raise SystemExit("no lure contracts classified")
    targets = sorted("0x" + a.lower()
                     for a in registry.load_token_registry(args.chain).legit_addresses)

    bucket_of = depth_buckets(args.chain, lure, targets)
    print(f"  depth buckets {DEPTH_CUTOFFS[:-1]}+: "
          f"{ {i: sum(1 for v in bucket_of.values() if v == i) for i in range(len(DEPTH_CUTOFFS))} }")
    probes = probe_addresses(PROBE_SEED)
    bare_to_addr = {bare(a): a for a in targets}

    # Enumerate row groups. Reading footers is cheap and gives an exact, balanced task list.
    tasks: list[tuple[str, int]] = []
    for cd in sorted((RAW / args.chain).glob("chunk_*")):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        tasks += [(str(lp), i) for i in range(pq.ParquetFile(lp).num_row_groups)]
    if not tasks:
        raise SystemExit(f"no completed chunks under {(RAW / args.chain).relative_to(ROOT)}")
    if partial:
        tasks = tasks[:args.max_row_groups]

    print(f"  {args.chain}: {len(lure)} lure contracts, {len(targets)} registry targets, "
          f"{len(probes)} probe addresses")
    print(f"  {len(tasks):,} row groups across {workers} workers"
          + ("  [PARTIAL PILOT]" if partial else ""))

    acc = {"rows": 0, "matched": 0, "probe_hits": 0,
           "per_contract": Counter(), "named_as_from": Counter(),
           "named_as_from_self": Counter(), "own_logs": Counter(),
           "own_logs_self_named": Counter(),
           # A LIST of per-task arrays, unioned once after the scan; see merge().
           "lure_recipients": [], "lure_senders": Counter(),
           "recips_by_bucket": {}, "rows_by_bucket": Counter(),
           "crit_rows": 0, "crit_contracts": Counter(), "crit_recipients": []}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(lure, targets, probes, args.batch_size, bucket_of)) as ex:
        for res in ex.map(_scan_row_group, tasks, chunksize=4):
            merge(acc, res)
            done += 1
            if done % 250 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = acc["rows"] / el if el else 0
                eta = (len(tasks) - done) / (done / el) if done and el else 0
                print(f"    {done:,}/{len(tasks):,} row groups, {acc['rows']:,} rows, "
                      f"{acc['matched']:,} lure rows, {rate/1e6:.1f}M rows/s, "
                      f"ETA {eta/60:.1f} min", flush=True)
    el = time.time() - t0

    # One union over all per-task arrays. Each task already de-duplicated its own rows, so this is
    # a single sort over the sum of per-task distinct counts rather than over every matched row.
    t_u = time.time()
    recipients = (np.unique(np.concatenate(acc["lure_recipients"]))
                  if acc["lure_recipients"] else np.empty(0, dtype="S40"))
    acc["lure_recipients"] = recipients
    union_s = time.time() - t_u
    if len(recipients):
        print(f"  recipient union: {len(recipients):,} distinct in {union_s:.1f}s")

    crit_recips = int(len(np.unique(np.concatenate(acc["crit_recipients"])))) \
        if acc["crit_recipients"] else 0

    # Cumulative depth table: buckets partition the population, so depth d unions buckets 0..i.
    depth_rows, cum = [], []
    for i, cut in enumerate(DEPTH_CUTOFFS):
        parts = acc["recips_by_bucket"].get(i, [])
        if parts:
            cum.extend(parts)
        distinct = int(len(np.unique(np.concatenate(cum)))) if cum else 0
        rows_cum = sum(acc["rows_by_bucket"].get(j, 0) for j in range(i + 1))
        contracts_cum = sum(1 for v in bucket_of.values() if v <= i)
        depth_rows.append({
            "cutoff": (None if cut >= 10**9 else cut),
            "cutoff_label": ("all" if cut >= 10**9 else str(cut)),
            "contracts": contracts_cum,
            "transfers": rows_cum,
            "distinct_recipients": distinct,
            "marginal_contracts": contracts_cum - (depth_rows[-1]["contracts"] if depth_rows else 0),
            "marginal_transfers": rows_cum - (depth_rows[-1]["transfers"] if depth_rows else 0),
            "marginal_recipients": distinct - (depth_rows[-1]["distinct_recipients"] if depth_rows else 0),
        })
    print(f"  fabricated-sender criterion: {acc['crit_rows']:,} rows "
          f"({acc['crit_rows']/max(1,acc['matched'])*100:.2f}%), "
          f"{len(acc['crit_contracts'])} contracts, {crit_recips:,} recipients")
    print("  depth table (cumulative):")
    for d in depth_rows:
        print(f"    <= {d['cutoff_label']:>4s}: contracts {d['contracts']:>4d} "
              f"(+{d['marginal_contracts']:>3d})  transfers {d['transfers']:>12,} "
              f"recipients {d['distinct_recipients']:>11,} (+{d['marginal_recipients']:>10,})")

    if not acc["matched"]:
        if not partial:
            raise SystemExit("zero lure rows matched on a FULL pass -- refusing to emit a census "
                             "that would read as 'no channel'. Check the 0x prefix on the metadata "
                             "addresses (INTEGRATION_NOTES 3b).")
        print("  NOTE: no lure rows in this pilot window. Expected when piloting the earliest row "
              "groups, since the campaigns are recent. Not evidence of anything.")

    # A fan-out below 1 is arithmetically impossible (each transfer names exactly one recipient), so
    # it would mean the recipient union over-counted -- the failure mode a parallel merge invites.
    n_r = len(acc["lure_recipients"])
    if n_r and acc["matched"] / n_r < 1.0:
        raise SystemExit(f"fan-out {acc['matched'] / n_r:.4f} < 1 with {acc['matched']:,} transfers "
                         f"and {n_r:,} distinct recipients -- impossible; the recipient union is "
                         f"wrong, not the data.")

    # Re-key from the bare join form back to 0x-prefixed addresses for the output.
    per_contract = {("0x" + k): v for k, v in acc["per_contract"].items()}
    named = {bare_to_addr.get(k, "0x" + k): v for k, v in acc["named_as_from"].items()}
    named_self = {bare_to_addr.get(k, "0x" + k): v for k, v in acc["named_as_from_self"].items()}
    own = {bare_to_addr.get(k, "0x" + k): v for k, v in acc["own_logs"].items()}
    own_self = {bare_to_addr.get(k, "0x" + k): v for k, v in acc["own_logs_self_named"].items()}

    ranked = sorted(named.items(), key=lambda kv: -kv[1])
    doc = {
        "chain": args.chain,
        "scope": ("FULL CHAIN -- every row group scanned" if not partial
                  else f"PARTIAL -- first {args.max_row_groups} row groups only, NOT publishable"),
        "partial": partial,
        "rows_scanned": acc["rows"], "lure_rows_matched": acc["matched"],
        "row_groups": len(tasks), "workers": workers, "wall_seconds": round(el, 1),
        "rows_per_second": round(acc["rows"] / el) if el else None,
        "match_key": ("low 20 bytes of the address and of each participant topic, compared as "
                      "40-character lowercase hex; independent of topic padding width"),
        "probe": {
            "seed": PROBE_SEED, "addresses": probes,
            "occurrences_in_chain": acc["probe_hits"],
            "absent": acc["probe_hits"] == 0,
            "interpretation": ("the ledger test in forgery_probe.py requires these addresses to be "
                               "unknown to the chain; occurrences_in_chain == 0 establishes that by "
                               "measurement rather than by appeal to improbability. Counted over "
                               "both participant fields of every row."),
        },
        "channel_census": {
            "transfers": acc["matched"],
            "distinct_recipients": int(len(acc["lure_recipients"])),
            "fanout_ratio": (acc["matched"] / len(acc["lure_recipients"])
                             if len(acc["lure_recipients"]) else None),
            "distinct_named_senders": len(acc["lure_senders"]),
            "top_named_senders": [{"address": "0x" + a, "transfers": n}
                                  for a, n in acc["lure_senders"].most_common(50)],
            "top_fifty_share": (sum(n for _, n in acc["lure_senders"].most_common(50))
                                / acc["matched"] if acc["matched"] else None),
            "note": ("recipients are addresses NAMED as recipient, excluding the zero address; "
                     "fan-out ~1 is the broadcast signature. Senders are the adversary-authored "
                     "field and identify no operator."),
        },
        "fabricated_sender_criterion": {
            "definition": ("a broadcast transfer whose named sender is a registry-verified token; "
                           "evidence of fabricated provenance, independent of the payload strings "
                           "the detector reads"),
            "rows": acc["crit_rows"],
            "row_share": (acc["crit_rows"] / acc["matched"]) if acc["matched"] else None,
            "contracts": len(acc["crit_contracts"]),
            "contract_share": (len(acc["crit_contracts"]) / len(per_contract)) if per_contract else None,
            "recipients": crit_recips,
            "recipient_share": (crit_recips / len(recipients)) if len(recipients) else None,
            "contracts_list": ["0x" + a for a in sorted(acc["crit_contracts"])],
        },
        "depth_table": depth_rows,
        "depth_cutoffs": [None if c >= 10**9 else c for c in DEPTH_CUTOFFS],
        "depth_note": ("cumulative at each rank cutoff; buckets partition the flagged population by "
                       "the emitting contract's volume rank, so recipients at depth d are the union "
                       "over buckets up to d. Marginal columns are the increment each interval adds."),
        "lure_transfers_by_contract": dict(sorted(per_contract.items(), key=lambda kv: -kv[1])),
        "lure_contracts_observed": len(per_contract),
        "lure_contracts_expected": len(lure),
        "impersonation": {
            "note": ("named_as_from counts rows asserting the target as sender. emitted_by_self "
                     "counts those the target itself emitted. A forged sender field predicts "
                     "emitted_by_self == 0."),
            "targets": [
                {"address": a, "named_as_from": n,
                 "emitted_by_self": named_self.get(a, 0),
                 "own_logs": own.get(a, 0),
                 "own_logs_naming_self": own_self.get(a, 0)}
                for a, n in ranked
            ],
        },
        "totals": {
            "rows_naming_a_registry_token_as_sender": sum(named.values()),
            "of_which_emitted_by_that_token": sum(named_self.values()),
            "registry_own_logs": sum(own.values()),
            "registry_own_logs_naming_self": sum(own_self.values()),
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"{name}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/forgery_chain_scan.py",
        "git_commit": git_commit(), "workers": workers,
    }, indent=1) + "\n")

    print(f"\n  probe addresses found in chain: {acc['probe_hits']} "
          f"({'absent as required' if not acc['probe_hits'] else 'PRESENT -- ledger test invalid'})")
    cc = doc["channel_census"]
    print(f"  lure contracts observed {len(per_contract)}/{len(lure)}")
    if cc["fanout_ratio"] is not None:
        print(f"  channel: {cc['transfers']:,} transfers to {cc['distinct_recipients']:,} "
              f"distinct named recipients (fan-out {cc['fanout_ratio']:.4f}); "
              f"{cc['distinct_named_senders']:,} named senders, top-50 "
              f"{cc['top_fifty_share']*100:.2f}%")
    t = doc["totals"]
    print(f"  rows naming a registry token as sender: "
          f"{t['rows_naming_a_registry_token_as_sender']:,}; "
          f"emitted by that token: {t['of_which_emitted_by_that_token']:,}")
    print(f"  registry tokens' own logs: {t['registry_own_logs']:,}; "
          f"naming themselves: {t['registry_own_logs_naming_self']:,}")
    print(f"  -> {out.relative_to(ROOT)}  ({el/60:.1f} min, "
          f"{acc['rows']/el/1e6:.1f}M rows/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
