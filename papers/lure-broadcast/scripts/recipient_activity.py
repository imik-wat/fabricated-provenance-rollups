"""Estimate the share of named recipients that are addresses in independent use.

Purpose
-------
This script addresses the distance between what the corpus measures and what the phenomenon is
claimed to be. A count of transfer events naming an address establishes a broadcast pattern; it does
not establish that any user was reached. Nothing in these records can show that a person viewed a
wallet interface, and no such claim is made here. What can be determined is the intermediate
question: whether the addresses the channel names are wallets in independent use, or addresses with
no independent activity.

Choice of observable: why "named as sender" will not do
-------------------------------------------------------
The first version of this script called an address active when it appeared in the sender field of
any Transfer event not emitted by a lure contract. That is invalid, and for the reason this paper
itself establishes: the sender field is written by the emitting contract and authenticated by
nothing, so an arbitrary contract can place any address there. Using it as evidence of key control
would contradict the paper's own central finding. The non-lure restriction does not repair it,
because the flagged set is a lower bound drawn from the volume head, so the unflagged remainder
certainly contains fabricating contracts.

The observable used instead is anchored to contracts whose implementation is independently verified,
the same registry the fabricated-sender criterion uses. For a verified ERC-20, a Transfer event
carrying a non-zero value is emitted only by `transfer`, where the sender field is the caller, or by
`transferFrom`, which requires an allowance that the named address granted. Either way the named
address must have signed something. The non-zero restriction is what excludes the zero-value
`transferFrom`, which the standard permits without an allowance and which is the mechanism of the
zero-value transfer attack; under it the sender field names an address that took no action at all.

Two consequences are stated rather than hidden. The estimate is a LOWER BOUND on activity, because
an address transacting only in tokens outside the registry is not seen; and the registry is small,
being the major tokens of each chain. Both apply identically to the recipients and to the control
group, so the contrast between them is unaffected even though each level is understated. For
comparison the unauthenticated quantity is reported alongside, labelled as such: the gap between the
two measures how much apparent activity rests on contract assertions rather than on signatures.

Quantities estimated, for lure recipients and for a control group the channel never named:

  ACTIVE (verified)   the address is the sender of at least one NON-ZERO transfer emitted by a
                      registry-verified token contract. This requires a signature or an allowance
                      from that address under the verified implementation.
  PRIOR-ACTIVE        that first verified send precedes the address's first lure delivery, so the
                      address was already in use when the channel reached it.
  ACTIVE (asserted)   the address appears in the sender field of any non-lure contract's Transfer.
                      Reported for contrast ONLY. It does not establish control and is never used
                      for a claim about who was reached.

Neither quantity measures intent, identity or loss, and no such claim is derived from this output.
`p_prior_active` bears on targeting, which is a property of the address list, and on nothing else.

Design
------
The estimand is per DISTINCT ADDRESS, so sampling rows would be wrong: a row sample over-represents
recipients that were named many times, which is precisely the population feature under study. The
sample is therefore drawn on the address itself, by a deterministic keyed hash:

    addr is in the probe set  <=>  splitmix64(fnv1a(all 40 hex characters)) & (K-1) == 0

Three properties matter. It is an equal-probability sample of the distinct-address space, because
the fold is injective on 40-byte inputs and splitmix64 is a bijection whose low bits are uniform. It
requires no prior knowledge of the population, so a single pass suffices rather than a
discover-then-measure pair. It is a deterministic function of the address, so every occurrence of a
given address is included or excluded together, which is the condition for tracking an address
across the pass.

The hash covers the WHOLE address, and the first version of this script did not: it mixed the last
eight characters only, which is enough for uniformity over unstructured addresses but collapses on
structured ones. Vanity generators mine addresses sharing a suffix, so under that version every
member of a vanity family was sampled or skipped TOGETHER -- a cluster, not an equal-probability
sample, and clustered on a trait correlated with being active. A test that builds a population
sharing a suffix and demands the nominal rate is what caught it, and that test is retained.

The sampler is not trusted. `n_sampled_recipients * K` is checked against the INDEPENDENTLY computed
census total (campaign_census_<chain>.json) and the run RAISES if the two disagree by more than six
binomial standard errors. A biased or broken hash cannot pass that check quietly.

Inputs
------
data/raw/hypersync/<chain>/chunk_*/logs.parquet   (read-only, ALL row groups)
analysis/config/token_metadata.json
papers/lure-broadcast/results/campaign_census_<chain>.json   (for the sampler check)

Outputs
-------
papers/lure-broadcast/results/recipient_activity_<chain>.json (+ .manifest.json)

Failure modes
-------------
- zero lure rows matched                      -> RAISE (an empty result is not a null result)
- a topic column not exactly 66 characters    -> RAISE (the buffer arithmetic below assumes it)
- a topic that is NULL                        -> excluded and COUNTED, never silently dropped
- fewer than MIN_SAMPLED sampled recipients   -> RAISE (an interval nobody should quote)
- sampler check outside six standard errors   -> RAISE (the hash, not the data, is then the finding)

Non-indexed Transfer events
---------------------------
ERC-20 requires `from` and `to` to be indexed, so they arrive as topics. A small number of contracts
emit the event with NO indexed parameters, putting both addresses in `data` instead; the topics are
then null. Those rows cannot be attributed to an address by the topic arithmetic here, so they are
excluded and their count is reported in `non_indexed_rows`. They are a handful per chain against
billions, but reporting the number is the difference between a known exclusion and a silent one --
and the same rows are the reason a null topic is handled rather than assumed away.

Complexity: one pass, four columns projected. Per row: one uint64 view and eight integer ops. Memory
is bounded by the sampled fraction, not by the chain: survivors are reduced to distinct addresses
within each batch before being retained.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402
from src import registry as reg  # noqa: E402
from src.dataio import chunk_dirs  # noqa: E402
from src.stats import wilson  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

TOPIC_CHARS = 66          # "0x" + 64 hex; the corpus is a topic0-filtered Transfer pull
ADDR_LO, ADDR_HI = 26, 66  # the 40 hex characters of an address inside a 32-byte topic
ADDR_CHARS = ADDR_HI - ADDR_LO           # 40 characters == 5 uint64 words, all of them hashed
ADDR_WORDS = ADDR_CHARS // 8
ZERO40 = b"0" * 40
MIN_SAMPLED = 1000        # below this the Wilson interval is too wide to be worth publishing
SAMPLER_SIGMA = 6.0       # tolerance on the census cross-check, in binomial standard errors

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Avalanche a uint64 vector. Bijective, so distinct inputs stay distinct.

    The inputs here are ASCII hex characters packed eight to a word, which occupy a sparse and
    highly structured corner of the 64-bit space; the point of the mixer is that the low bits of the
    output are uniform regardless.
    """
    z = (x + _GOLDEN).astype(np.uint64)
    z = ((z ^ (z >> _S30)) * _M1).astype(np.uint64)
    z = ((z ^ (z >> _S27)) * _M2).astype(np.uint64)
    return (z ^ (z >> _S31)).astype(np.uint64)


def fold_address(words: np.ndarray) -> np.ndarray:
    """Fold the five uint64 words of a 40-character address into one, FNV-1a style.

    Injective in practice on the inputs that occur: each word is eight ASCII hex characters, and the
    multiply-xor chain is the standard construction whose collisions over five words are not
    reachable from a 2^160 address space at these population sizes. Every character participates, so
    no family of addresses sharing a prefix or a suffix is sampled as a block.
    """
    h = np.full(words.shape[0], _FNV_OFFSET, dtype=np.uint64)
    for j in range(words.shape[1]):
        h = ((h ^ words[:, j]) * _FNV_PRIME).astype(np.uint64)
    return h


def as_char_matrix(col: pa.Array, width: int) -> tuple[np.ndarray, np.ndarray]:
    """View a fixed-width string column as an (n, width) uint8 matrix, plus a validity mask.

    Nulls are filled with a placeholder so the buffer stays rectangular and the row indices keep
    lining up with the block numbers; the mask, not the placeholder, is what the caller must use.
    Returning the mask rather than dropping the rows is deliberate: a dropped row shifts every
    index after it, which is how an off-by-one becomes a plausible-looking proportion.

    RAISES if any value has a different width, because every offset computed downstream assumes it.
    """
    a = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(a)
    if n == 0:
        return np.empty((0, width), dtype=np.uint8), np.empty(0, dtype=bool)
    valid = np.ones(n, dtype=bool)
    if a.null_count:
        valid = ~np.asarray(pc.is_null(a).to_numpy(zero_copy_only=False), dtype=bool)
        a = pc.fill_null(a, "0" * width)
        if isinstance(a, pa.ChunkedArray):
            a = a.combine_chunks()
    offs = np.frombuffer(a.buffers()[1], dtype=np.int32, count=n + 1, offset=a.offset * 4)
    if not np.all(np.diff(offs) == width):
        raise SystemExit(f"topic column is not uniformly {width} characters wide")
    data = np.frombuffer(a.buffers()[2], dtype=np.uint8)
    return data[offs[0]:offs[0] + width * n].reshape(n, width), valid


def sample_mask(chars: np.ndarray, k_mask: np.uint64) -> np.ndarray:
    """Which rows' addresses fall in the probe set. `chars` is the (n, 66) topic matrix."""
    if chars.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    words = np.ascontiguousarray(chars[:, ADDR_LO:ADDR_HI]).view(np.uint64).reshape(-1, ADDR_WORDS)
    return (splitmix64(fold_address(words)) & k_mask) == 0


def reduce_batch(chars: np.ndarray, keep: np.ndarray, blocks: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Distinct sampled addresses in this batch, with the earliest block each was seen at.

    Reducing inside the batch is what bounds memory: the retained arrays grow with the number of
    distinct sampled addresses on the chain, not with the number of rows.
    """
    if not keep.any():
        return np.empty((0, 40), dtype=np.uint8), np.empty(0, dtype=np.int64)
    sub = np.ascontiguousarray(chars[keep][:, ADDR_LO:ADDR_HI])
    blk = blocks[keep]
    uniq, inv = np.unique(sub.view(np.dtype((np.void, 40))).ravel(), return_inverse=True)
    first = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inv, blk)
    return uniq.view(np.uint8).reshape(-1, 40), first


class FirstSeen:
    """Accumulate (address -> earliest block) across batches, reducing only at the end."""

    def __init__(self) -> None:
        self._addrs: list[np.ndarray] = []
        self._blocks: list[np.ndarray] = []

    def add(self, addrs: np.ndarray, blocks: np.ndarray) -> None:
        if len(addrs):
            self._addrs.append(addrs)
            self._blocks.append(blocks)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._addrs:
            return np.empty(0, dtype=np.dtype((np.void, 40))), np.empty(0, dtype=np.int64)
        a = np.concatenate(self._addrs).view(np.dtype((np.void, 40))).ravel()
        b = np.concatenate(self._blocks)
        uniq, inv = np.unique(a, return_inverse=True)
        first = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, inv, b)
        return uniq, first


def drop_zero(addrs: np.ndarray, zero, blocks: np.ndarray | None = None):
    """Remove the zero address, which is a mint or burn counterparty and not an account."""
    if len(addrs) == 0:
        return addrs, blocks
    keep = addrs != zero
    return addrs[keep], (blocks[keep] if blocks is not None else None)


def index_in(needles: np.ndarray, haystack: np.ndarray) -> np.ndarray:
    """Position of each needle in a SORTED haystack, or -1 where absent.

    Both arguments are `np.unique` output, hence sorted and distinct, so a binary search is exact.
    A dict keyed on the addresses would be the obvious alternative and does not work: numpy's
    writeable void scalars are unhashable, and making them hashable means materialising millions of
    Python objects for no gain.
    """
    if len(haystack) == 0 or len(needles) == 0:
        return np.full(len(needles), -1, dtype=np.int64)
    pos = np.clip(np.searchsorted(haystack, needles), 0, len(haystack) - 1)
    return np.where(haystack[pos] == needles, pos, -1)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def lure_contracts(chain: str) -> dict[str, str]:
    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[chain]
    out: dict[str, str] = {}
    for a, m in md.items():
        v = ml.classify(m.get("name"), m.get("symbol"))
        if v.strength == "strong":
            k = ml.campaign_key(v)
            if k:
                out[a.lower()] = k
    return out


def proportion(hits: int, n: int) -> dict[str, float]:
    """A count with its Wilson interval. An empty group yields None, never a zero-width interval."""
    if n <= 0:
        return {"hits": hits, "n": n, "estimate": None, "ci95_lo": None, "ci95_hi": None}
    lo, hi = wilson(hits, n)
    return {"hits": hits, "n": n, "estimate": hits / n if n else None,
            "ci95_lo": lo, "ci95_hi": hi}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--k", type=int, default=1024,
                    help="sample one address in K; must be a power of two")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.k < 2 or args.k & (args.k - 1):
        raise SystemExit("--k must be a power of two (the mask below is a bitwise and)")
    k_mask = np.uint64(args.k - 1)

    out = OUT_ROOT / f"recipient_activity_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    campaign_of = lure_contracts(args.chain)
    if not campaign_of:
        raise SystemExit(f"no lure contracts on {args.chain}")
    lure_arr = pa.array(sorted(campaign_of))

    # The verified-token registry anchors the authenticated observable. A registry contract's
    # implementation is independently verified, so its sender field is determined by the standard
    # rather than asserted at will.
    verified = sorted("0x" + a.lower().removeprefix("0x")
                      for a in reg.load_token_registry(args.chain).legit_addresses)
    if not verified:
        raise SystemExit(f"no registry-verified tokens for {args.chain}; the authenticated "
                         f"observable has no anchor and the run would silently fall back to the "
                         f"contract-asserted one")
    ver_arr = pa.array(verified)
    print(f"  {args.chain}: {len(campaign_of)} lure contracts, {len(verified)} verified tokens, "
          f"sampling 1 address in {args.k}")

    lure_recv, gen_send, gen_recv = FirstSeen(), FirstSeen(), FirstSeen()
    ver_send, ver_recv = FirstSeen(), FirstSeen()
    rows = matched = non_indexed = verified_rows = 0
    t0 = time.time()
    for cd in chunk_dirs(RAW / args.chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        for b in pq.ParquetFile(lp).iter_batches(
                batch_size=1_000_000,
                columns=["address", "topic1", "topic2", "block_number", "data"]):
            rows += b.num_rows
            blocks = b.column("block_number").to_numpy(zero_copy_only=False).astype(np.int64)
            addr = pc.utf8_lower(b.column("address"))
            is_lure = np.asarray(
                pc.is_in(addr, value_set=lure_arr).to_numpy(zero_copy_only=False), dtype=bool)
            matched += int(is_lure.sum())

            # Registry-verified emitter carrying a non-zero value word. Trimming the "0" characters
            # from the data payload leaves an empty string exactly when every digit is zero, which
            # is the zero-value transferFrom the standard permits without an allowance.
            is_ver = np.asarray(
                pc.is_in(addr, value_set=ver_arr).to_numpy(zero_copy_only=False), dtype=bool)
            body = pc.utf8_slice_codeunits(b.column("data"), 2, 2 ** 30)
            nonzero = np.asarray(
                pc.greater(pc.utf8_length(pc.utf8_trim(body, characters="0")), 0)
                .to_numpy(zero_copy_only=False), dtype=bool)
            is_ver_nz = is_ver & nonzero
            verified_rows += int(is_ver_nz.sum())

            t1, ok1 = as_char_matrix(b.column("topic1"), TOPIC_CHARS)
            t2, ok2 = as_char_matrix(b.column("topic2"), TOPIC_CHARS)
            non_indexed += int((~(ok1 & ok2)).sum())
            s1 = sample_mask(t1, k_mask) & ok1
            s2 = sample_mask(t2, k_mask) & ok2

            # The lure contracts' recipient list; the authenticated observable; and the
            # contract-asserted one, retained only for the contrast between them.
            lure_recv.add(*reduce_batch(t2, s2 & is_lure, blocks))
            ver_send.add(*reduce_batch(t1, s1 & is_ver_nz, blocks))
            ver_recv.add(*reduce_batch(t2, s2 & is_ver_nz, blocks))
            gen_send.add(*reduce_batch(t1, s1 & ~is_lure, blocks))
            gen_recv.add(*reduce_batch(t2, s2 & ~is_lure, blocks))
        print(f"    {cd.name}: {rows:,} rows, {matched:,} lure, {time.time()-t0:.0f}s", flush=True)

    el = time.time() - t0
    if not matched:
        raise SystemExit("zero lure rows matched -- refusing to emit an empty result")

    r_addr, r_first = lure_recv.finish()
    vs_addr, vs_first = ver_send.finish()      # authenticated: verified emitter, non-zero value
    vr_addr, _ = ver_recv.finish()
    s_addr, _ = gen_send.finish()              # contract-asserted, for contrast only
    v_addr, _ = gen_recv.finish()

    # The zero address is not an account. A mint emits Transfer(0x0, to, value) with a non-zero
    # value from a registry token, so without this it would enter the verified-sender set and be
    # counted as an active address in the comparison group. It is excluded from every accumulator,
    # not only from the recipient list. (At K=1024 the zero address does not fall in the probe set
    # at all, so the published estimates are unaffected; the guard is unconditional regardless.)
    zero = np.frombuffer(ZERO40, dtype=np.uint8).view(np.dtype((np.void, 40)))[0]
    r_addr, r_first = drop_zero(r_addr, zero, r_first)
    vs_addr, vs_first = drop_zero(vs_addr, zero, vs_first)
    vr_addr, _ = drop_zero(vr_addr, zero)
    s_addr, _ = drop_zero(s_addr, zero)
    v_addr, _ = drop_zero(v_addr, zero)
    n_r = len(r_addr)
    if n_r < MIN_SAMPLED:
        raise SystemExit(f"only {n_r} sampled recipients (< {MIN_SAMPLED}); lower --k and rerun")

    # Cross-check the sampler against the independently computed census.
    census_path = OUT_ROOT / f"campaign_census_{args.chain}.json"
    sampler_check = None
    if census_path.exists():
        n_pop = json.loads(census_path.read_text())["channel_totals"]["distinct_recipients_union"]
        expected = n_pop / args.k
        sigma = np.sqrt(n_pop * (1.0 / args.k) * (1 - 1.0 / args.k))
        dev = (n_r - expected) / sigma
        sampler_check = {"census_distinct_recipients": n_pop, "expected_sampled": expected,
                         "observed_sampled": n_r, "deviation_sigma": float(dev)}
        if abs(dev) > SAMPLER_SIGMA:
            raise SystemExit(
                f"sampler check FAILED: {n_r} sampled recipients against {expected:.0f} expected "
                f"from the census ({dev:+.1f} sigma). The hash, not the channel, is the finding.")

    # Authenticated observable.
    vsent = index_in(r_addr, vs_addr)
    has_vsent = vsent >= 0
    vblock = np.where(has_vsent, vs_first[np.maximum(vsent, 0)], np.iinfo(np.int64).max)
    prior = has_vsent & (vblock < r_first)
    vreceived = index_in(r_addr, vr_addr) >= 0

    # Contract-asserted observable, reported only for the contrast with the above.
    asserted = index_in(r_addr, s_addr) >= 0

    # Control: sampled addresses observed anywhere in the corpus that no lure contract named. The
    # control is formed from the same universe under both observables so the contrast is internal.
    seen = np.union1d(np.union1d(s_addr, v_addr), np.union1d(vs_addr, vr_addr))
    ctrl = seen[index_in(seen, r_addr) < 0]
    ctrl_vsent = index_in(ctrl, vs_addr) >= 0
    ctrl_asserted = index_in(ctrl, s_addr) >= 0

    doc = {
        "chain": args.chain,
        "estimand": ("per DISTINCT ADDRESS named as a recipient by a lure contract: the share that "
                     "independently sent a transfer emitted by a non-lure contract (ACTIVE), and "
                     "the share whose first such send precedes their first lure delivery "
                     "(PRIOR-ACTIVE). Neither is intent, identity or financial loss."),
        "sampling": {"scheme": "splitmix64(fnv1a(all 40 hex characters of the address)) & (K-1) == 0",
                     "k": args.k, "unit": "distinct address, not transfer row",
                     "why_not_rows": ("a row sample over-represents repeatedly named recipients, "
                                      "which is the population feature under study")},
        "rows_scanned": rows, "rows_matched": matched, "wall_seconds": round(el, 1),
        "non_indexed_rows": non_indexed,
        "non_indexed_note": ("ERC-20 requires from/to to be indexed. These rows carry a Transfer "
                             "topic0 with null topics, both addresses being in `data` instead, so "
                             "they cannot be attributed to an address here and are excluded. "
                             "Counted rather than dropped."),
        "lure_contracts": len(campaign_of),
        "sampler_check": sampler_check,
        "sampled_recipients": n_r,
        "verified_rows_scanned": verified_rows,
        "observable": {
            "authenticated": ("sender of a NON-ZERO transfer emitted by a registry-verified token "
                              "contract; under the verified implementation this requires a "
                              "signature from the address or an allowance it granted"),
            "asserted": ("named in the sender field of any non-lure contract's transfer; this "
                         "establishes nothing about control and is reported only for contrast"),
            "verified_tokens": len(verified),
            "lower_bound_note": ("an address transacting only in tokens outside the registry is "
                                 "not observed, so each level understates activity; the same "
                                 "understatement applies to the control group, so the contrast "
                                 "between them is unaffected"),
        },
        "active": proportion(int(has_vsent.sum()), n_r),
        "prior_active": proportion(int(prior.sum()), n_r),
        "received_verified": proportion(int(vreceived.sum()), n_r),
        "active_asserted": proportion(int(asserted.sum()), n_r),
        "control": {
            "definition": ("sampled addresses observed in the corpus that no lure contract ever "
                           "named; the base rate of activity in the address population under the "
                           "same observable"),
            "active": proportion(int(ctrl_vsent.sum()), len(ctrl)),
            "active_asserted": proportion(int(ctrl_asserted.sum()), len(ctrl)),
        },
        "caveats": [
            "ACTIVE is evidence that the address signed or authorised something under a verified "
            "token's implementation. It is not evidence that a person read a wallet interface.",
            "The asserted observable is NOT evidence of control: the sender field of an arbitrary "
            "contract's transfer event is written by that contract. It is reported only so the "
            "distance between the two observables is visible.",
            "An address active only AFTER its first lure delivery is counted ACTIVE but not "
            "PRIOR-ACTIVE; the ordering is what distinguishes targeting from coincidence.",
            "The control group is addresses SEEN in the corpus, so it excludes addresses that never "
            "transacted; the comparison is against transacting addresses, not against all of "
            "2^160.",
            "ERC-721 sends count as activity: they equally require the private key.",
        ],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"recipient_activity_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/recipient_activity.py",
        "git_commit": git_commit(), "k": args.k,
    }, indent=1) + "\n")

    a, p, c = doc["active"], doc["prior_active"], doc["control"]["active"]
    aa, ca = doc["active_asserted"], doc["control"]["active_asserted"]
    print(f"\n  sampled recipients      {n_r:,}   (verified rows {verified_rows:,})")
    print(f"  active (verified)       {a['estimate']:.4f}  [{a['ci95_lo']:.4f}, {a['ci95_hi']:.4f}]")
    print(f"  prior-active            {p['estimate']:.4f}  [{p['ci95_lo']:.4f}, {p['ci95_hi']:.4f}]")
    print(f"  control (verified)      {c['estimate']:.4f}  [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]"
          f"  (n={c['n']:,})")
    print(f"  active (asserted)       {aa['estimate']:.4f}   control {ca['estimate']:.4f}"
          f"   <- NOT evidence of control")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
