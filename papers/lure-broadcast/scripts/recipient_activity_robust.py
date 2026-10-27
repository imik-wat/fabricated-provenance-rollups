"""Robustness of the recipient-activity contrast (IEEE review R2.7; plan WP8).

Reviewer 2 asked us to clarify the 7.98% vs 4.30% contrast on Base: its observational interpretation,
registry assumptions, temporal windows and confounders, without causal claims. The published
comparison has two weaknesses this script measures rather than argues:
  * TIME. "Ever sent a registry-listed transfer" is compared between named recipients and every
    sampled address the channel never named, at any time. The groups need not have existed over the
    same windows.
  * CONTAMINATION. The control is defined by the flagged set, which is a lower bound. Addresses that
    other broadcast or fabricating contracts named sit in the control group.

Variants, all on the PUBLISHED deterministic address sample (same hash, same K), so V1 must reproduce
the published counts exactly. The script raises otherwise:
  V1  published: ever registry-active, recipients vs never-named sampled addresses.
  V2  time-aligned. Each recipient's reference block is its first delivery, and each control
      address gets a reference block drawn, by a hash of the address itself, from the recipients'
      first-delivery distribution. Both groups are restricted to addresses already seen in non-lure
      rows before their reference block, and activity counts only before it. This removes
      differences due to when an address appeared; it does not remove selection on activity.
  V3  cleaned control: never-named addresses that no weak-band frame contract, no fabricating
      unflagged frame contract (frame_probe) and no flagged tail contract (tail_sample) named.
Each variant reports both proportions with Wilson intervals, the difference with Newcombe's
interval, and the ratio.

Interpretation, fixed in advance:
  * descriptive associations under the registry restriction, with the same pseudo-randomisation
    assumption as the published intervals;
  * activity shows that control over an address was exercised, never that anyone saw a payload;
  * no variant supports a causal statement about the channel.

Output: papers/lure-broadcast/results/recipient_activity_robust_<chain>.json (+ .manifest.json)
Complexity: one full-chain pass, the same cost as recipient_activity.py.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recipient_activity as ra  # noqa: E402
from src import metadata_lure as ml, registry as reg  # noqa: E402
from src.dataio import chunk_dirs  # noqa: E402
from src.stats import wilson  # noqa: E402

ROOT = ra.ROOT
OUT_ROOT = ra.OUT_ROOT


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def newcombe(h1: int, n1: int, h2: int, n2: int) -> dict:
    """Difference of two proportions with Newcombe's hybrid-score interval (method 10)."""
    if not n1 or not n2:
        return {"difference": None, "ci95_lo": None, "ci95_hi": None, "ratio": None}
    p1, p2 = h1 / n1, h2 / n2
    l1, u1 = wilson(h1, n1)
    l2, u2 = wilson(h2, n2)
    d = p1 - p2
    return {"difference": d,
            "ci95_lo": d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            "ci95_hi": d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
            "ratio": (p1 / p2) if p2 else None}


def address_uniform(addrs: np.ndarray) -> np.ndarray:
    """A deterministic uniform [0, 1) per address, from the HIGH bits of the published hash.

    The sampling decision uses only the low bits (hash & (K - 1) == 0), so the high bits are free to
    act as an independent draw.
    """
    if len(addrs) == 0:
        return np.empty(0)
    words = np.ascontiguousarray(addrs.view(np.uint8).reshape(-1, 40)).view(np.uint64).reshape(-1, ra.ADDR_WORDS)
    h = ra.splitmix64(ra.fold_address(words))
    return (h >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def reference_blocks(ctrl: np.ndarray, recipient_first: np.ndarray) -> np.ndarray:
    """Each control address's reference block, drawn from the recipients' first-delivery blocks."""
    ordered = np.sort(recipient_first)
    idx = np.minimum((address_uniform(ctrl) * len(ordered)).astype(np.int64), len(ordered) - 1)
    return ordered[idx]


def first_seen(addrs: np.ndarray, *sources: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Earliest block each address appears at in any source; int64 max where it never does."""
    out = np.full(len(addrs), np.iinfo(np.int64).max, dtype=np.int64)
    for s_addr, s_first in sources:
        pos = ra.index_in(addrs, s_addr)
        hit = pos >= 0
        out[hit] = np.minimum(out[hit], s_first[pos[hit]])
    return out


def time_aligned(ref: np.ndarray, seen: np.ndarray, active_first: np.ndarray) -> tuple[int, int]:
    """(hits, n): addresses already seen before `ref`, and those registry-active before it."""
    eligible = seen < ref
    return int((eligible & (active_first < ref)).sum()), int(eligible.sum())


ZERO_VOID = np.frombuffer(ra.ZERO40, dtype=np.uint8).view(np.dtype((np.void, 40)))[0]


def finish(acc: "ra.FirstSeen") -> tuple[np.ndarray, np.ndarray]:
    """Distinct addresses and first blocks from an accumulator, zero address removed.

    `ra.drop_zero` takes (addrs, zero, blocks). An earlier version unpacked `acc.finish()` straight
    into it, which passed the blocks where the zero marker belongs. That failed only after a full
    40-minute pass, so the call now sits here, where a unit test reaches it.
    """
    addrs, first = acc.finish()
    return ra.drop_zero(addrs, ZERO_VOID, first)


def other_broadcasters(chain: str) -> list[str]:
    """Contracts other than the flagged set whose recipients could contaminate the control."""
    raw = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[chain]
    out = {a.lower() for a, m in raw.items()
           if ml.classify(m.get("name"), m.get("symbol")).strength == "weak"}
    fp = OUT_ROOT / f"frame_probe_{chain}.json"
    if fp.exists():
        out |= set(json.loads(fp.read_text()).get("fabricated_not_flagged", []))
    tail = OUT_ROOT / f"tail_sample_{chain}.json"
    if tail.exists():
        out |= {r["address"] for r in json.loads(tail.read_text()).get("flagged_in_sample", [])}
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = OUT_ROOT / f"recipient_activity_robust_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")
    published = json.loads((OUT_ROOT / f"recipient_activity_{args.chain}.json").read_text())
    k_mask = np.uint64(args.k - 1)

    campaign_of = ra.lure_contracts(args.chain)
    lure_arr = pa.array(sorted(campaign_of))
    verified = sorted("0x" + a.lower().removeprefix("0x")
                      for a in reg.load_token_registry(args.chain).legit_addresses)
    ver_arr = pa.array(verified)
    others = other_broadcasters(args.chain)
    other_arr = pa.array(others)

    lure_recv, gen_send, gen_recv = ra.FirstSeen(), ra.FirstSeen(), ra.FirstSeen()
    ver_send, ver_recv, other_recv = ra.FirstSeen(), ra.FirstSeen(), ra.FirstSeen()
    t0 = time.time()
    for cd in chunk_dirs(ra.RAW / args.chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        for b in pq.ParquetFile(lp).iter_batches(
                batch_size=1_000_000, columns=["address", "topic1", "topic2", "block_number", "data"]):
            blocks = b.column("block_number").to_numpy(zero_copy_only=False).astype(np.int64)
            addr = pc.utf8_lower(b.column("address"))
            is_lure = np.asarray(pc.is_in(addr, value_set=lure_arr).to_numpy(zero_copy_only=False), dtype=bool)
            is_other = np.asarray(pc.is_in(addr, value_set=other_arr).to_numpy(zero_copy_only=False), dtype=bool)
            is_ver = np.asarray(pc.is_in(addr, value_set=ver_arr).to_numpy(zero_copy_only=False), dtype=bool)
            body = pc.utf8_slice_codeunits(b.column("data"), 2, 2 ** 30)
            nonzero = np.asarray(pc.greater(pc.utf8_length(pc.utf8_trim(body, characters="0")), 0)
                                 .to_numpy(zero_copy_only=False), dtype=bool)
            is_ver_nz = is_ver & nonzero
            t1, ok1 = ra.as_char_matrix(b.column("topic1"), ra.TOPIC_CHARS)
            t2, ok2 = ra.as_char_matrix(b.column("topic2"), ra.TOPIC_CHARS)
            s1 = ra.sample_mask(t1, k_mask) & ok1
            s2 = ra.sample_mask(t2, k_mask) & ok2
            lure_recv.add(*ra.reduce_batch(t2, s2 & is_lure, blocks))
            ver_send.add(*ra.reduce_batch(t1, s1 & is_ver_nz, blocks))
            ver_recv.add(*ra.reduce_batch(t2, s2 & is_ver_nz, blocks))
            gen_send.add(*ra.reduce_batch(t1, s1 & ~is_lure, blocks))
            gen_recv.add(*ra.reduce_batch(t2, s2 & ~is_lure, blocks))
            other_recv.add(*ra.reduce_batch(t2, s2 & is_other, blocks))
        print(f"    {cd.name}: {time.time() - t0:.0f}s", flush=True)

    r_addr, r_first = finish(lure_recv)
    vs_addr, vs_first = finish(ver_send)
    vr_addr, vr_first = finish(ver_recv)
    s_addr, s_first = finish(gen_send)
    v_addr, v_first = finish(gen_recv)
    o_addr, _ = finish(other_recv)

    # V1: the published comparison, reproduced exactly or the run stops.
    r_active_first = first_seen(r_addr, (vs_addr, vs_first))
    seen_union = np.union1d(np.union1d(s_addr, v_addr), np.union1d(vs_addr, vr_addr))
    ctrl = seen_union[ra.index_in(seen_union, r_addr) < 0]
    c_active_first = first_seen(ctrl, (vs_addr, vs_first))
    big = np.iinfo(np.int64).max
    v1 = {"recipients": ra.proportion(int((r_active_first < big).sum()), len(r_addr)),
          "control": ra.proportion(int((c_active_first < big).sum()), len(ctrl))}
    pa_, pc_ = published["active"], published["control"]["active"]
    if (v1["recipients"]["hits"], v1["recipients"]["n"], v1["control"]["hits"], v1["control"]["n"]) != (
            pa_["hits"], pa_["n"], pc_["hits"], pc_["n"]):
        raise SystemExit(f"V1 does not reproduce the published counts: {v1} vs {pa_}, {pc_}")
    v1["difference"] = newcombe(v1["recipients"]["hits"], v1["recipients"]["n"],
                                v1["control"]["hits"], v1["control"]["n"])

    # V2: time-aligned. "Seen" means appearing in any non-lure row.
    nonlure = ((s_addr, s_first), (v_addr, v_first))
    r_seen = first_seen(r_addr, *nonlure)
    c_seen = first_seen(ctrl, *nonlure)
    c_ref = reference_blocks(ctrl, r_first)
    rh, rn = time_aligned(r_first, r_seen, r_active_first)
    ch, cn = time_aligned(c_ref, c_seen, c_active_first)
    v2 = {"recipients": ra.proportion(rh, rn), "control": ra.proportion(ch, cn),
          "difference": newcombe(rh, rn, ch, cn)}

    # V3: control without addresses that other broadcasters or fabricators named.
    clean = ra.index_in(ctrl, o_addr) < 0
    v3 = {"recipients": v1["recipients"],
          "control": ra.proportion(int(((c_active_first < big) & clean).sum()), int(clean.sum())),
          "control_removed": int((~clean).sum()), "other_contracts": len(others)}
    v3["difference"] = newcombe(v3["recipients"]["hits"], v3["recipients"]["n"],
                                v3["control"]["hits"], v3["control"]["n"])

    doc = {"chain": args.chain, "k": args.k, "V1_published": v1, "V2_time_aligned": v2,
           "V3_cleaned_control": v3,
           "interpretation": ("descriptive associations on the registry-restricted observable, "
                              "with pseudo-randomisation intervals as published; activity shows "
                              "exercised control over an address, not that anyone saw a payload; "
                              "no variant supports a causal statement"),
           "wall_seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/recipient_activity_robust.py",
        "git_commit": git_commit(), "k": args.k}, indent=1) + "\n")
    for name in ("V1_published", "V2_time_aligned", "V3_cleaned_control"):
        d = doc[name]
        print(f"  {name:20s} recipients {d['recipients']['estimate']:.4f} (n={d['recipients']['n']})  "
              f"control {d['control']['estimate']:.4f} (n={d['control']['n']})  "
              f"diff {d['difference']['difference']:+.4f} [{d['difference']['ci95_lo']:+.4f}, "
              f"{d['difference']['ci95_hi']:+.4f}]  ratio {d['difference']['ratio']:.2f}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
