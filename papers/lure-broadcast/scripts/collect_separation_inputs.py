"""Collect, in one scan, everything the target-list overlap test needs on Base.

What this produces
------------------
  family A   -- per-campaign recipient lists for the metadata-lure contracts (48 campaigns on Base)
  family A'  -- CALIBRATION: per-contract recipient lists for high-fan-out contracts the lure detector
                scores `none`, i.e. legitimate mass distributors. Mandatory, not optional: without it an
                O/E above 1 cannot be distinguished from strata that are simply too coarse. Synthetic
                validation showed a pure popularity confounder producing O/E = 2.23 under coarse strata.
  family B   -- poisoning target lists, from the published symmetric rule at tau >= 4. No payout label is
                consulted, so nothing here restates a result of the manuscript under review.
  universe   -- the eligible address population, with a stratum label per address.

Why the universe is a SAMPLED sub-corpus, and why that is sound here
-------------------------------------------------------------------
The strata need per-address aggregates (activity class, era), and Base carries ~3.23 billion
registry-token rows across tens of millions of addresses. A per-address hash map over that does not fit in
memory, so the scan samples row groups stratified across every chunk and defines the universe as the
addresses observed in that sample.

This is safe for THIS statistic in a way it was not for the per-chain shares. Row-group sampling is
time-stratified sampling in disguise, which biased a share by 3.34 pp -- but here both families AND the
stratum pools are drawn from the SAME sampled row groups, so the quantity reported is a ratio whose arms
share the sampling frame. The scope is stated in the output rather than implied: the claim is about the
sampled sub-corpus, and the sampled fraction is recorded.

Why the universe must be BROADER than the poisoning support
----------------------------------------------------------
The universe is registry-token recipients, NOT the candidate-generating support. On Base that support is
86-99% victims, so family B's list would be nearly the whole universe and the overlap would be forced --
the same degeneracy that killed the marginal ratio, one level up. Broadening the universe is what leaves
the null somewhere to move; `nullmodel`'s saturation and no-freedom guards catch it if this is got wrong.

Strata
------
Built only from covariates that are neither descendants of the exposure nor colliders on it (see
docs/INTEGRATION_NOTES.md and the banned list below):
  - `sent`  : did the address ever SEND a registry token with amount > 0 in the sample. The one
              unimpeachable proxy for a real controlled wallet -- nothing an attacker does to you can make
              you send.
  - `act`   : count of registry-token rows involving the address, bucketed. The mechanical driver of both
              attractiveness and collision probability.
  - `era`   : first block seen, in deciles of the sampled block range.
BANNED and not computed: total transfers received (a collider on exposure and latent activity), unfiltered
sent counts (the ZTT label puts the victim on the `from` side, so they are outcome-contaminated), and
non-fungible receipt count (that IS the exposure for the other channel).

Inputs
------
data/raw/hypersync/base/chunk_*/logs.parquet   (read-only)
analysis/config/token_metadata.json            (lure classification)
results/base_ztt_candidates.csv                (poisoning victims, read-only)

Outputs
-------
papers/lure-broadcast/results/separation_inputs_base.json.gz (+ .manifest.json)

Failure modes
-------------
- A registry join that matches nothing RAISES: `registry.legit_addresses` has no `0x` prefix and a direct
  join silently matches zero rows (see INTEGRATION_NOTES 3b), which would read as "no activity".
- An empty calibration family RAISES: the run is not interpretable without it.
- Every family list is intersected with the universe and the loss is reported, never silently dropped.

Complexity: one pass over the sampled row groups. At 10 row groups per chunk Base reads ~100M rows in a
few minutes; memory is dominated by the per-address aggregate map.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml, registry  # noqa: E402
from src.dataio import ERC20_DATA_HEXLEN, ZERO_DATA, chunk_dirs  # noqa: E402
from src.heuristic import ZERO_ADDRESS  # noqa: E402

RAW = ROOT / "data" / "raw" / "hypersync"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
#: Fan-out floor for a contract to count as a mass distributor, from Forta's published rule.
FANOUT_MIN = 499
ACT_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 9), (10, 24), (25, 99), (100, 499), (500, 10 ** 12))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _addr(topic: str) -> str:
    return "0x" + topic[-40:]


def act_bucket(n: int) -> str:
    for lo, hi in ACT_BUCKETS:
        if lo <= n <= hi:
            return f"a{lo}"
    return "a1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--per-chunk", type=int, default=10, help="row groups sampled per chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"separation_inputs_{args.chain}.json.gz"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    # ---- classify the cached metadata into lure campaigns and calibration distributors ----------
    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text()).get(args.chain)
    if not md:
        raise SystemExit(f"no cached metadata for {args.chain}")
    campaign_of: dict[str, str] = {}
    none_contracts: set[str] = set()
    for a, m in md.items():
        v = ml.classify(m.get("name"), m.get("symbol"))
        if v.strength == "strong":
            k = ml.campaign_key(v)
            if k:
                campaign_of[a.lower()] = k
        elif v.strength == "none":
            none_contracts.add(a.lower())

    # NOTE the "0x" re-prefix: registry.legit_addresses stores addresses WITHOUT it, and a direct join
    # matches nothing silently (INTEGRATION_NOTES 3b). This is the detect.py:232 idiom.
    legit = sorted("0x" + a for a in registry.load_token_registry(args.chain).legit_addresses)
    if not legit:
        raise SystemExit("empty registry")
    legit_arr = pa.array(legit)
    lure_arr = pa.array(sorted(campaign_of))
    none_arr = pa.array(sorted(none_contracts))
    print(f"  {args.chain}: {len(legit)} registry tokens, {len(campaign_of)} lure contracts in "
          f"{len(set(campaign_of.values()))} campaigns, {len(none_contracts)} candidate calibration "
          f"contracts")

    # ---- one sampled scan -----------------------------------------------------------------------
    rng = random.Random(args.seed)
    groups: list[tuple[Path, int]] = []
    for cd in chunk_dirs(RAW / args.chain):
        lp = cd / "logs.parquet"
        if not lp.exists() or not (cd / "_DONE").exists():
            continue
        n = pq.ParquetFile(lp).num_row_groups
        groups.extend((lp, i) for i in sorted(rng.sample(range(n), min(args.per_chunk, n))))
    if not groups:
        raise SystemExit("no sampled row groups")

    campaign_recipients: dict[str, set[str]] = defaultdict(set)
    calib_recipients: dict[str, set[str]] = defaultdict(set)
    act: dict[str, int] = defaultdict(int)
    first_block: dict[str, int] = {}
    sent_value: set[str] = set()
    universe: set[str] = set()
    rows = legit_rows = 0
    t0 = time.time()
    cols = ["address", "data", "topic1", "topic2", "block_number"]
    for lp, gi in groups:
        t = pq.ParquetFile(lp).read_row_group(gi, columns=cols)
        rows += t.num_rows
        addr = pc.utf8_lower(t.column("address"))
        lens = pc.utf8_length(t.column("data"))
        is20 = pc.equal(lens, ERC20_DATA_HEXLEN)

        # registry-token rows -> universe, activity, era, sender flag
        sub = t.filter(pc.and_(is20, pc.is_in(addr, value_set=legit_arr)))
        if sub.num_rows:
            legit_rows += sub.num_rows
            f = sub.column("topic1").to_pylist()
            r = sub.column("topic2").to_pylist()
            d = sub.column("data").to_pylist()
            b = sub.column("block_number").to_pylist()
            for fi, ri, di, bi in zip(f, r, d, b):
                if not fi or not ri:
                    continue
                s, rc = _addr(fi), _addr(ri)
                for x in (s, rc):
                    if x == ZERO_ADDRESS:
                        continue
                    act[x] += 1
                    if x not in first_block or bi < first_block[x]:
                        first_block[x] = bi
                if rc != ZERO_ADDRESS:
                    universe.add(rc)
                if di != ZERO_DATA and s != ZERO_ADDRESS:
                    sent_value.add(s)

        # lure-contract rows -> per-campaign recipient lists
        sub = t.filter(pc.is_in(addr, value_set=lure_arr))
        if sub.num_rows:
            a_ = pc.utf8_lower(sub.column("address")).to_pylist()
            r_ = sub.column("topic2").to_pylist()
            for ai, ri in zip(a_, r_):
                if ri:
                    campaign_recipients[campaign_of[ai]].add(_addr(ri))

        # calibration candidates -> per-contract recipient lists
        sub = t.filter(pc.is_in(addr, value_set=none_arr))
        if sub.num_rows:
            a_ = pc.utf8_lower(sub.column("address")).to_pylist()
            r_ = sub.column("topic2").to_pylist()
            for ai, ri in zip(a_, r_):
                if ri:
                    calib_recipients[ai].add(_addr(ri))
    el = time.time() - t0
    if not legit_rows:
        raise SystemExit("zero registry-token rows matched -- almost certainly the 0x-prefix join bug "
                         "(INTEGRATION_NOTES 3b); refusing to continue with an empty universe")
    print(f"  scanned {len(groups)} row groups, {rows:,} rows in {el:.0f}s; "
          f"{legit_rows:,} registry rows, universe {len(universe):,} addresses")

    # ---- strata ---------------------------------------------------------------------------------
    blocks = sorted(first_block.values())
    deciles = [blocks[int(q * (len(blocks) - 1))] for q in (0.1 * i for i in range(1, 10))]

    def era(b: int) -> int:
        return sum(1 for d in deciles if b > d)

    strata = {x: f"{'S' if x in sent_value else 'R'}|{act_bucket(act[x])}|e{era(first_block[x])}"
              for x in universe}
    # Store the RAW covariates too, not just the derived label. Stratum granularity is the knob most
    # likely to need sweeping -- the calibration arm exists precisely to detect when it is too coarse --
    # and storing only the label would force a full re-scan for every refinement.
    covariates = {x: [act[x], first_block[x], 1 if x in sent_value else 0] for x in universe}

    # ---- family B: poisoning victims under the published rule ------------------------------------
    victims: set[str] = set()
    cpath = ROOT / "results" / f"{args.chain}_ztt_candidates.csv"
    if not cpath.exists():
        raise SystemExit(f"missing {cpath.relative_to(ROOT)}")
    with cpath.open() as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row["sim_prefix"]) >= 4 and int(row["sim_suffix"]) >= 4:
                    victims.add(row["victim"].lower())
            except (KeyError, ValueError):
                continue

    # Keep only the top calibration distributors by fan-out, matching the lure family's scale.
    calib = {k: v for k, v in calib_recipients.items() if len(v) >= FANOUT_MIN}
    if not calib:
        raise SystemExit(f"no calibration distributor reached fan-out {FANOUT_MIN} in the sample; the "
                         f"run is not interpretable without a calibration arm -- raise --per-chunk")

    def restrict(fam: dict[str, set[str]]) -> tuple[dict[str, list[str]], dict]:
        kept, lost_in, lost_out = {}, 0, 0
        for k, s in fam.items():
            inside = s & universe
            lost_in += len(s) - len(inside)
            lost_out += len(s)
            if inside:
                kept[k] = sorted(inside)
        return kept, {"addresses_before": lost_out, "addresses_outside_universe": lost_in}

    fam_a, loss_a = restrict(campaign_recipients)
    fam_cal, loss_cal = restrict(calib)
    vic_in = sorted(victims & universe)

    doc = {
        "chain": args.chain,
        "scope": ("A sampled sub-corpus: row groups drawn per chunk. Both families and the stratum pools "
                  "come from the SAME sampled row groups, so the reported quantity is a ratio whose arms "
                  "share the sampling frame."),
        "sampling": {"row_groups": len(groups), "per_chunk": args.per_chunk, "rows_read": rows,
                     "seed": args.seed, "wall_seconds": round(el, 1)},
        "universe_size": len(universe),
        "registry_rows": legit_rows,
        "strata": strata,
        "covariates": covariates,
        "covariate_schema": ["registry_row_count", "first_block_seen", "ever_sent_value"],
        "era_deciles": deciles,
        "family_lure": fam_a,
        "family_calibration": fam_cal,
        "poisoning_victims": vic_in,
        "poisoning_victims_total": len(victims),
        "losses": {"lure": loss_a, "calibration": loss_cal,
                   "victims_outside_universe": len(victims) - len(vic_in)},
        "banned_covariates": [
            "total transfers received -- collider on exposure and latent activity",
            "unfiltered sent counts -- the ZTT label puts the victim on the `from` side",
            "non-fungible receipt count -- that IS the other channel's exposure",
        ],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        json.dump(doc, fh)
    (OUT_ROOT / f"separation_inputs_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/collect_separation_inputs.py",
        "git_commit": git_commit(), "seed": args.seed, "fanout_min": FANOUT_MIN,
    }, indent=1) + "\n")

    print(f"  family_lure       {len(fam_a)} campaigns, "
          f"{sum(len(v) for v in fam_a.values()):,} in-universe targets")
    print(f"  family_calibration{len(fam_cal):>3} distributors, "
          f"{sum(len(v) for v in fam_cal.values()):,} in-universe targets")
    print(f"  poisoning victims {len(vic_in):,} of {len(victims):,} in universe "
          f"({len(vic_in)/max(len(victims),1)*100:.1f}%)")
    print(f"  strata            {len(set(strata.values()))} distinct")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
