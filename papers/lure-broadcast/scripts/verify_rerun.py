"""Re-run each committed producer and compare its output with the committed artefact.

Why this exists
---------------
Twenty-nine artefacts under papers/lure-broadcast/results/ were produced before papers/lure-broadcast/
was tracked in git. Their manifests name a `git_commit` (7420bfe, fbb8bd2 or cb1bddd) that does not
contain the producing script, so the manifest does not establish which code produced them. The
producers are now committed. Whether the committed code reproduces the committed artefact is a
question of measurement, and this script makes that measurement, one artefact at a time.

What it does, per artefact
--------------------------
1. RUN (only with --run). Invoke the committed producer with the parameters recorded in the
   artefact's manifest (defaults otherwise) and `--force`, so it overwrites the artefact in the
   working tree. The exact command line, return code and wall time are recorded.
2. COMPARE. Deep-compare the working-tree artefact and its manifest with `git show HEAD:<path>`:
   JSON recursively, CSV row by row. Every difference is reported as a path with its old and new
   value (capped at --max-diffs per file; the total count is always reported).
3. CLASSIFY. `identical` | `identical_except_volatile` | `differs` | `failed` | `timeout`.
4. RESTORE (only with --run). `git checkout HEAD -- <artefact> <manifest>` and every other tracked
   file the run modified, and delete any new untracked file the run created under the paper tree.
   Producers read each other's artefacts, so every later producer must run against committed
   inputs, never against a previous re-run's output.

The volatile rule
-----------------
A field may be ignored only if it is wall-clock time, a timestamp or date, a duration, a throughput
or a git commit. That rule is applied as the short, explicit allowlist `VOLATILE_KEYS`, matched
against the LAST component of a difference's path (a dict key). Nothing is ignored by value, by
tolerance or by pattern. Ignored differences are not discarded: they are listed, with old and new
values, under `ignored_fields`, so a reader can see what the verdict did not weigh.

Anything else that differs is a real difference -- including an RPC answer that changed, a coverage
count, an endpoint choice, an environment description or an ordering -- and is reported as one.

Isolation and side effects
--------------------------
Writes papers/lure-broadcast/results/verification_rerun.json, merging per artefact so artefacts can
be verified in separate invocations. Producer stdout/stderr go to --log-dir (outside the repository
by default). Before and after each run the set of content-modified tracked files and of untracked
files is snapshotted (`core.fileMode=false`, so mode-only changes on filesystems without POSIX modes
are not mistaken for edits); everything the run changed is recorded under `side_effects` and
restored. Untracked files created outside the paper tree are recorded but never deleted.

Because of that restore, `--run` must not share a checkout with anyone else's edits: a file another
session changes while a producer runs is indistinguishable from a side effect and would be reverted.
Producers also write shared caches (analysis/config/token_metadata.json, the code cache). `--run`
therefore refuses to start outside a linked worktree (`git worktree add`) unless
--allow-main-checkout is passed; comparing alone is always safe.

Inputs
------
papers/lure-broadcast/scripts/<producer>.py   (the committed producers, run unmodified)
git HEAD                                      (the committed artefacts, via `git show`)
whatever each producer reads; untracked inputs a producer needs are declared per spec in
`untracked_inputs` and recorded with their resolved location, because a reproduction that depends
on a file outside version control is weaker than one that does not.

Outputs
-------
papers/lure-broadcast/results/verification_rerun.json

Stamping manifests
------------------
`--stamp RECORD [RECORD ...]` writes a `reproduced` block -- the commit the record ran at, its verdict and the
ignored fields -- into the manifest of every artefact RECORD found reproduced, so a reader of the
manifest sees which commit containing the producer reproduces it. It RAISES when the producer, a
lure-broadcast script it imports (transitively), the shared library (analysis/src) or the committed
artefact changed between that commit and HEAD, since
the stamp would then describe code or data the run never saw. The stamp is this script's own output,
not the producer's, so comparison drops it from both manifests before diffing. With several records,
a later one supersedes an earlier one for the artefacts it covers.

Portable records
----------------
A record ships with the release, so it holds no machine paths: a path inside any checkout of the
repository (main or linked) is written repository-relative, any other absolute path is cut to
`<outside>/<name>`, and the interpreter is written as `python`. Writing a record RAISES if an
absolute path survives. `--portable RECORD ...` rewrites records written before this rule.

Usage
-----
    # run, compare and restore two artefacts, in order
    python scripts/verify_rerun.py --run --only polygon-zkevm_nf_census polygon-zkevm_token_volumes
    # compare the working tree with HEAD for every artefact, running nothing
    python scripts/verify_rerun.py
    # record the reproduction in each reproduced artefact's manifest
    python scripts/verify_rerun.py --stamp results/verification_rerun.json

Failure modes
-------------
- A producer exiting non-zero      -> `failed`, with the tail of its stderr. Never compared.
- A producer exceeding --timeout   -> its whole process group is killed; `timeout`.
- A producer exiting 0 without rewriting the artefact (mtime unchanged) -> `failed`: comparing an
  untouched file with HEAD would report `identical` for a run that measured nothing.
- An artefact missing from HEAD    -> raises; there is nothing to compare against.
- A malformed artefact             -> `failed` with the parse error, never a silent pass.

Limitations
-----------
- `identical` means byte-for-value identical after JSON/CSV parsing; key order and whitespace are
  not compared (all producers write sorted keys).
- The verdict covers the artefact and its manifest. It does not re-derive anything independently:
  it tests that the committed code, on today's inputs and today's network, gives the committed
  answer.
- Producers that call public RPC endpoints are re-measured against the network as it is now. A
  difference there may be a changed answer rather than a code change; the record marks such
  producers `network: true` so the reading is not conflated.

Complexity: dominated by the producers (full-chain passes take minutes to tens of minutes); the
comparison itself is linear in artefact size.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
RES = PAPER / "results"
REL_RES = "papers/lure-broadcast/results"
OUT = RES / "verification_rerun.json"

#: The ONLY keys whose differences may be ignored. Each is a wall-clock duration, a throughput
#: (or its inverse, seconds per item) or a git commit, per the volatile rule above. Matched exactly
#: against the last dict key of a difference's path; extend only with a field of those kinds.
VOLATILE_KEYS: frozenset[str] = frozenset({
    "git_commit",                       # provenance commit
    "wall_seconds",                     # duration
    "total_wall_seconds",               # duration (sum of durations)
    "canary_seconds",                   # duration of an endpoint canary
    "rows_per_second",                  # throughput
    "peak_rows_per_second",             # throughput
    "seconds_per_contract",             # inverse throughput
    "metadata_seconds_per_contract",    # inverse throughput
    "fastest_seconds_per_contract",     # inverse throughput
    "slowest_seconds_per_contract",     # inverse throughput
})

#: The manifest key --stamp writes; never written by a producer.
STAMP_KEY = "reproduced"
REPRODUCED = ("identical", "identical_except_volatile")
#: Code a stamped artefact depends on besides its producer.
SHARED_CODE = "analysis/src"

DEFAULT_MAX_DIFFS = 25
DEFAULT_TIMEOUT_S = 3 * 3600
_ABSENT = "<absent>"
#: Keys rendered with dot notation; anything else (URLs, domains, spaces) is quoted in brackets.
_IDENT = re.compile(r"^[A-Za-z0-9_\-]+$")


# ------------------------------------------------------------------------------------ specs
@dataclass(frozen=True)
class Spec:
    """One artefact and the invocation that should reproduce it.

    `artefact` is the file name under results/; `manifest` is derived from it. `args` are passed to
    the producer verbatim after the script path. `untracked_inputs` are repository-relative paths the
    producer reads that are not under version control. `network` marks producers whose output
    depends on live RPC answers.
    """
    artefact: str
    producer: str
    args: tuple[str, ...]
    untracked_inputs: tuple[str, ...] = ()
    network: bool = False

    @property
    def stem(self) -> str:
        return self.artefact.rsplit(".", 1)[0]

    @property
    def manifest(self) -> str:
        return f"{self.stem}.manifest.json"


def _raw(chain: str) -> str:
    return f"data/raw/hypersync/{chain}"


def _vol(chain: str) -> str:
    return f"results/{chain}_token_volumes.csv"


def build_specs() -> list[Spec]:
    """The 29 artefacts, upstream first. Parameters are those recorded in each manifest (workers,
    top, k, classes); everything else is the producer's default. `--force` is always passed."""
    chains5 = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]
    s: list[Spec] = []
    for c in chains5:
        s.append(Spec(f"{c}_nf_census.json", "run_nf_census.py", (c, "--force"), (_raw(c),)))
    s.append(Spec("polygon-zkevm_token_volumes.csv", "token_volumes.py",
                  ("polygon-zkevm", "--force"), (_raw("polygon-zkevm"),)))
    for c in chains5:
        w = "4" if c == "zksync-era" else "12"
        vol = () if c == "polygon-zkevm" else (_vol(c),)   # polygon's table is in the paper tree
        s.append(Spec(f"forgery_chain_{c}.json", "forgery_chain_scan.py",
                      ("--chain", c, "--workers", w, "--force"), (_raw(c),) + vol))
    s.append(Spec("campaign_census_base.json", "campaign_census.py", ("--chain", "base", "--force"),
                  (_raw("base"),)))
    for c in ("arbitrum-one", "base", "optimism"):
        s.append(Spec(f"residual_probe_{c}.json", "residual_probe.py", ("--chain", c, "--force"),
                      (_raw(c),)))
    s.append(Spec("forgery_probe_base.json", "forgery_probe.py", ("--chain", "base", "--force"),
                  (_raw("base"),), network=True))
    s.append(Spec("interface_probe_base.json", "interface_probe.py",
                  ("--chain", "base", "--top", "200", "--force"), (_raw("base"),), network=True))
    for c in ("arbitrum-one", "base", "optimism"):
        s.append(Spec(f"recipient_activity_{c}.json", "recipient_activity.py",
                      ("--chain", c, "--k", "1024", "--force"), (_raw(c),)))
    s.append(Spec("residue_robustness_base.json", "residue_robustness.py",
                  ("--chain", "base", "--k", "1024", "--classes", "8", "--force"), (_raw("base"),)))
    s.append(Spec("residue_robustness_polygon-zkevm.json", "residue_robustness.py",
                  ("--chain", "polygon-zkevm", "--k", "64", "--classes", "4", "--force"),
                  (_raw("polygon-zkevm"),)))
    s.append(Spec("metadata_history_base.json", "metadata_history.py", ("--chain", "base", "--force"),
                  (_raw("base"),), network=True))
    s.append(Spec("unresolved_profile_base.json", "unresolved_profile.py",
                  ("--chain", "base", "--force"), (), network=True))
    for c in ("arbitrum-one", "optimism", "polygon-zkevm", "zksync-era"):
        vol = () if c == "polygon-zkevm" else (_vol(c),)
        s.append(Spec(f"metadata_sweep_{c}.json", "run_metadata_sweep.py", ("--chain", c, "--force"),
                      vol, network=True))
    s.append(Spec("systems.json", "systems_report.py", ("--force",), ("data/raw/hypersync",)))
    return s


SPECS: list[Spec] = build_specs()


# ------------------------------------------------------------------------------- comparison
def fmt_path(parent: str, key: Any) -> str:
    """Render one path step: `.key` for identifier-like keys, `["key"]` otherwise, `[i]` for lists."""
    if isinstance(key, int):
        return f"{parent}[{key}]"
    k = str(key)
    if _IDENT.match(k):
        return f"{parent}.{k}" if parent else k
    return f'{parent}["{k}"]'


def _same_leaf(a: Any, b: Any) -> bool:
    """Exact equality for JSON leaves. A type change is a difference (1 vs 1.0, True vs 1); two NaNs
    are equal, since JSON round-trips them identically."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def json_diff(old: Any, new: Any, path: str = "", last_key: str | None = None) -> list[dict]:
    """Every leaf-level difference between two parsed JSON documents.

    Returns a list of {"path", "old", "new", "volatile"} in deterministic (sorted-key) order. A key
    present on one side only is reported with the absent side as "<absent>"; a list length change
    is reported at the list's path and the surplus items individually. `volatile` is True when the
    last dict key on the path is in VOLATILE_KEYS -- the subtree below a volatile key inherits it.
    Pure; O(size of the documents).
    """
    vol = last_key in VOLATILE_KEYS
    out: list[dict] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            p = fmt_path(path, k)
            if k not in old:
                out.append({"path": p, "old": _ABSENT, "new": new[k],
                            "volatile": vol or k in VOLATILE_KEYS})
            elif k not in new:
                out.append({"path": p, "old": old[k], "new": _ABSENT,
                            "volatile": vol or k in VOLATILE_KEYS})
            else:
                sub = json_diff(old[k], new[k], p, k if not vol else last_key)
                out.extend(sub)
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append({"path": f"{path}.length" if path else "length", "old": len(old),
                        "new": len(new), "volatile": vol})
        for i in range(max(len(old), len(new))):
            p = fmt_path(path, i)
            if i >= len(new):
                out.append({"path": p, "old": old[i], "new": _ABSENT, "volatile": vol})
            elif i >= len(old):
                out.append({"path": p, "old": _ABSENT, "new": new[i], "volatile": vol})
            else:
                out.extend(json_diff(old[i], new[i], p, last_key))
        return out
    if not _same_leaf(old, new):
        out.append({"path": path or "<root>", "old": old, "new": new, "volatile": vol})
    return out


def csv_diff(old_text: str, new_text: str) -> list[dict]:
    """Row-wise CSV comparison. Row 0 is the header. Never volatile: no CSV column here is a time.

    Reports a row-count change at `rows.length`, then each differing row index as `row[i]` with the
    old and new row (absent rows as "<absent>"). Pure; O(rows).
    """
    o = list(csv.reader(io.StringIO(old_text)))
    n = list(csv.reader(io.StringIO(new_text)))
    out: list[dict] = []
    if len(o) != len(n):
        out.append({"path": "rows.length", "old": len(o), "new": len(n), "volatile": False})
    for i in range(max(len(o), len(n))):
        a = o[i] if i < len(o) else _ABSENT
        b = n[i] if i < len(n) else _ABSENT
        if a != b:
            out.append({"path": f"row[{i}]", "old": a, "new": b, "volatile": False})
    return out


def compare_text(name: str, old_text: str, new_text: str) -> list[dict]:
    """Dispatch on extension. JSON parse errors propagate (the caller turns them into `failed`)."""
    if name.endswith(".csv"):
        return csv_diff(old_text, new_text)
    return json_diff(json.loads(old_text), json.loads(new_text))


def unstamped(manifest_text: str) -> Any:
    """A manifest without the `reproduced` stamp, which this script writes and a producer never does."""
    doc = json.loads(manifest_text)
    if isinstance(doc, dict):
        doc.pop(STAMP_KEY, None)
    return doc


def classify(real: list[dict], volatile: list[dict]) -> str:
    """Verdict for a completed run. Pure."""
    if real:
        return "differs"
    return "identical_except_volatile" if volatile else "identical"


def split_diffs(diffs: list[dict]) -> tuple[list[dict], list[dict]]:
    real = [d for d in diffs if not d["volatile"]]
    vol = [d for d in diffs if d["volatile"]]
    return real, vol


def _strip(ds: list[dict], cap: int) -> list[dict]:
    return [{"path": d["path"], "old": d["old"], "new": d["new"]} for d in ds[:cap]]


# -------------------------------------------------------------------------------------- git
def git(*args: str, check: bool = True) -> str:
    """Run git against THIS repository (ROOT), never another checkout."""
    r = subprocess.run(["git", "-C", str(ROOT), "-c", "core.fileMode=false", *args],
                       capture_output=True, text=True, check=check)
    return r.stdout


def git_show_head(rel: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{rel} is not in HEAD ({r.stderr.strip()}); nothing to compare against")
    return r.stdout


def tree_state() -> tuple[set[str], set[str]]:
    """(content-modified tracked paths vs HEAD, untracked non-ignored paths)."""
    modified = set(filter(None, git("diff", "--name-only", "HEAD").splitlines()))
    untracked = set(filter(None, git("ls-files", "--others", "--exclude-standard").splitlines()))
    return modified, untracked


def is_linked_worktree() -> bool:
    """True when ROOT is a `git worktree add` checkout rather than the repository's main one."""
    gd = git("rev-parse", "--git-dir").strip()
    common = git("rev-parse", "--git-common-dir").strip()
    return (ROOT / gd).resolve() != (ROOT / common).resolve()


def restore(paths: list[str]) -> None:
    if paths:
        git("checkout", "HEAD", "--", *paths)


# -------------------------------------------------------------------------------------- run
@dataclass
class RunResult:
    returncode: int | None
    wall_seconds: float
    timed_out: bool
    log: str
    stderr_tail: list[str] = field(default_factory=list)


def run_producer(cmd: list[str], cwd: Path, log_path: Path, timeout_s: float) -> RunResult:
    """Run one producer in its own process group, so a timeout kills its worker pool too."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("$ " + " ".join(cmd) + f"\n# cwd {cwd}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc, timed_out = None, True
    el = time.time() - t0
    lines = log_path.read_text(errors="replace").splitlines()
    # Drop the harmless hashlib blake2 tracebacks the system OpenSSL emits on import.
    tail = [ln for ln in lines if "blake2" not in ln and "hashlib" not in ln][-25:]
    return RunResult(rc, round(el, 1), timed_out, str(log_path), tail)


def _mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def _resolve_input(rel: str) -> dict:
    p = ROOT / rel
    info: dict[str, Any] = {"path": rel, "exists": p.exists(), "symlink": p.is_symlink(),
                            "resolved": str(p.resolve())}
    if p.is_symlink():
        info["target"] = os.readlink(p)
    tracked = bool(git("ls-files", "--", rel).strip())
    info["tracked"] = tracked
    return info


def verify_one(spec: Spec, run: bool, python: str, log_dir: Path, timeout_s: float,
               max_diffs: int) -> dict:
    """Run (optionally), compare and restore one artefact. Returns its record."""
    art_rel, man_rel = f"{REL_RES}/{spec.artefact}", f"{REL_RES}/{spec.manifest}"
    art, man = ROOT / art_rel, ROOT / man_rel
    cmd = [python, f"scripts/{spec.producer}", *spec.args]
    rec: dict[str, Any] = {
        "artefact": art_rel, "manifest": man_rel,
        "producer": f"papers/lure-broadcast/scripts/{spec.producer}",
        "arguments": list(spec.args),
        "command": " ".join([python, f"scripts/{spec.producer}", *spec.args]),
        "cwd": str(PAPER.relative_to(ROOT)),
        "network": spec.network,
        "commit": git("rev-parse", "HEAD").strip(),
        "untracked_inputs": [_resolve_input(r) for r in spec.untracked_inputs],
    }
    old_art, old_man = git_show_head(art_rel), git_show_head(man_rel)
    rec["committed_manifest"] = json.loads(old_man)

    rec["mode"] = "run" if run else "compare_only"
    if not run:
        _compare_into(rec, spec, art, man, old_art, old_man, max_diffs)
        return rec

    before_mod, before_untr = tree_state()
    rec["tree_content_changes_at_start"] = sorted(before_mod)
    m0 = (_mtime(art), _mtime(man))
    t_start = time.time()
    rec["started_at_unix"] = round(t_start, 1)
    try:
        rr = run_producer(cmd, PAPER, log_dir / f"{spec.stem}.log", timeout_s)
        rec["run_wall_seconds"] = rr.wall_seconds
        rec["returncode"] = rr.returncode
        rec["log"] = rr.log
        if rr.timed_out:
            rec["verdict"] = "timeout"
            rec["error"] = f"exceeded {timeout_s:.0f} s; process group killed"
            rec["stderr_tail"] = rr.stderr_tail
        elif rr.returncode != 0:
            rec["verdict"] = "failed"
            rec["error"] = f"producer exited {rr.returncode}"
            rec["stderr_tail"] = rr.stderr_tail
        else:
            m1 = (_mtime(art), _mtime(man))
            rewritten = m1[0] is not None and m1[0] != m0[0] and m1[0] >= t_start - 2
            rec["artefact_rewritten"] = rewritten
            rec["manifest_rewritten"] = m1[1] is not None and m1[1] != m0[1]
            if rewritten:
                # Keep the re-run output outside the repository before it is restored away, so a
                # difference can be investigated beyond the listed paths.
                for src in (art, man):
                    if src.exists():
                        keep = log_dir / f"{src.stem}.rerun{src.suffix}"
                        keep.write_bytes(src.read_bytes())
                rec["rerun_copy"] = str(log_dir / f"{art.stem}.rerun{art.suffix}")
                _compare_into(rec, spec, art, man, old_art, old_man, max_diffs)
            else:
                rec["verdict"] = "failed"
                rec["error"] = "producer exited 0 but did not rewrite the artefact"
                rec["stderr_tail"] = rr.stderr_tail
    finally:
        # Whatever happened -- success, failure, timeout or an interrupt -- restore the committed
        # artefact and manifest and undo every other change the run made to the tree.
        after_mod, after_untr = tree_state()
        side_tracked = sorted((after_mod - before_mod) - {art_rel, man_rel})
        side_untracked = sorted(after_untr - before_untr)
        rec["side_effects"] = {"tracked_modified": side_tracked,
                               "untracked_created": side_untracked}
        restore([art_rel, man_rel] + side_tracked)
        removed = []
        for u in side_untracked:
            if u.startswith("papers/lure-broadcast/"):
                (ROOT / u).unlink(missing_ok=True)
                removed.append(u)
        rec["restored"] = [art_rel, man_rel] + side_tracked
        rec["removed_untracked"] = removed
        now_mod = tree_state()[0]
        left = (now_mod - before_mod) | ({art_rel, man_rel} & now_mod)
        rec["clean_after_restore"] = not left
        if left:
            rec["not_restored"] = sorted(left)
    return rec


def _compare_into(rec: dict, spec: Spec, art: Path, man: Path, old_art: str, old_man: str,
                  max_diffs: int) -> None:
    try:
        d_art = compare_text(spec.artefact, old_art, art.read_text())
        d_man = json_diff(unstamped(old_man), unstamped(man.read_text()))
    except (OSError, ValueError) as e:
        rec["verdict"] = "failed"
        rec["error"] = f"could not compare: {type(e).__name__}: {e}"
        return
    real_a, vol_a = split_diffs(d_art)
    real_m, vol_m = split_diffs(d_man)
    for d in real_m + vol_m:
        d["path"] = "manifest:" + d["path"]
    real, vol = real_a + real_m, vol_a + vol_m
    rec["verdict"] = classify(real, vol)
    rec["differences_total"] = len(real)
    rec["differences"] = _strip(real, max_diffs)
    # Every ignored field is listed, uncapped: what the verdict did not weigh must stay visible.
    rec["ignored_fields"] = _strip(vol, len(vol))
    rec["ignored_fields_total"] = len(vol)


# ------------------------------------------------------------------------------------ stamp
def changed_since(commit: str, path: str) -> bool:
    """True if `path` (file or directory) differs between `commit` and HEAD."""
    r = subprocess.run(["git", "-C", str(ROOT), "-c", "core.fileMode=false", "diff", "--quiet",
                        commit, "HEAD", "--", path])
    if r.returncode not in (0, 1):
        raise SystemExit(f"git diff {commit} HEAD -- {path} failed (exit {r.returncode})")
    return r.returncode == 1


def local_deps(producer: str) -> list[str]:
    """The producer and every script beside it that it imports, transitively (repository-relative).

    A file-level dependency, deliberately: forgery_chain_scan imports two unchanged names from a
    forgery_probe whose RPC handling changed, and deciding that the change cannot matter is the kind
    of judgement a stamp must not rest on.
    """
    folder = Path(producer).parent
    seen: list[str] = []
    todo = [Path(producer).stem]
    while todo:
        rel = str(folder / f"{todo.pop()}.py")
        if rel in seen:
            continue
        seen.append(rel)
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            else:
                continue
            todo += [m.split(".")[0] for m in mods if (ROOT / folder / f"{m.split('.')[0]}.py").exists()]
    return seen


def stamp(record_paths: list[Path]) -> list[str]:
    """Write the `reproduced` stamp into the manifest of every artefact the records found reproduced
    by a run, a later record superseding an earlier one per artefact. Returns the stems stamped.
    Raises before writing anything if any of them depends on code or data that changed since its run."""
    latest: dict[str, tuple[str, dict]] = {}
    for path in record_paths:
        doc = load_record(path)
        if not doc.get("artefacts"):
            raise SystemExit(f"{path} holds no artefact records")
        rel = str(path.resolve().relative_to(ROOT.resolve()))
        latest.update({stem: (rel, rec) for stem, rec in doc["artefacts"].items()})
    todo = [(stem, rel, rec) for stem, (rel, rec) in latest.items()
            if rec.get("mode") == "run" and rec.get("verdict") in REPRODUCED]
    for stem, _, rec in todo:
        for path in [*local_deps(rec["producer"]), SHARED_CODE, rec["artefact"]]:
            if changed_since(rec["commit"], path):
                raise SystemExit(f"{stem}: {path} changed between {rec['commit'][:7]} and HEAD; the "
                                 f"stamp would describe a run that never saw it. Re-run first.")
    for stem, rel, rec in todo:
        man = ROOT / rec["manifest"]
        m = json.loads(man.read_text())
        m[STAMP_KEY] = {"commit": rec["commit"], "verdict": rec["verdict"],
                        "by": "papers/lure-broadcast/scripts/verify_rerun.py", "record": rel,
                        "ignored_fields": sorted({d["path"] for d in rec.get("ignored_fields", [])})}
        man.write_text(json.dumps(m, indent=1) + "\n")
    # An artefact whose latest run did not reproduce loses any earlier stamp: a stale claim of
    # reproduction is worse than none.
    stamped = {stem for stem, _, _ in todo}
    for stem, (_, rec) in latest.items():
        man = ROOT / rec["manifest"]
        if stem not in stamped and man.exists():
            m = json.loads(man.read_text())
            if m.pop(STAMP_KEY, None) is not None:
                man.write_text(json.dumps(m, indent=1) + "\n")
    return [stem for stem, _, _ in todo]


# ------------------------------------------------------------------------------------- main
ABS_PATH = re.compile(r"/(?:private|tmp|var|Users|Volumes|home|opt)/[^\s'\"]*")


def checkout_roots() -> list[str]:
    """Every checkout of this repository (main and linked worktrees), longest first, since a record
    written in a worktree names files in both."""
    listed = [ln.split(" ", 1)[1] for ln in git("worktree", "list", "--porcelain").splitlines()
              if ln.startswith("worktree ")]
    roots = {str(ROOT), str(ROOT.resolve()), *listed, *(str(Path(r).resolve()) for r in listed)}
    return sorted(roots, key=len, reverse=True)


def portable(value: Any, roots: list[str], python: str | None = None) -> Any:
    """`value` with every machine path removed (see "Portable records"). Recurses into dicts and
    lists; returns non-strings unchanged. Pure."""
    if isinstance(value, dict):
        return {k: portable(v, roots, python) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, roots, python) for v in value]
    if not isinstance(value, str):
        return value
    if python:
        value = value.replace(python, "python")
    for r in roots:
        value = value.replace(r + "/", "").replace(r, ".")
    return ABS_PATH.sub(lambda m: "<outside>/" + Path(m.group(0)).name, value)


def write_record(path: Path, doc: dict, python: str | None = None) -> None:
    """Write a record with machine paths removed; raise if one survives."""
    doc = portable(doc, checkout_roots(), python)
    text = json.dumps(doc, indent=1, sort_keys=False, default=str) + "\n"
    left = ABS_PATH.findall(text)
    if left:
        raise SystemExit(f"{path}: machine paths survive in the record: {left[:3]}")
    path.write_text(text)


def load_record(path: Path) -> dict:
    """The existing record, so per-artefact invocations accumulate. A malformed record RAISES rather
    than being silently replaced, which would discard earlier verdicts without a trace."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError as e:
        raise SystemExit(f"{path} exists but is not valid JSON ({e}); move it aside first")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true",
                    help="run each producer (with --force) before comparing, then restore")
    ap.add_argument("--only", nargs="+", default=None,
                    help="artefact stems to verify (default: all 29, upstream first)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the producers")
    ap.add_argument("--log-dir", default=str(Path(tempfile.gettempdir()) / "verify_rerun_logs"))
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--max-diffs", type=int, default=DEFAULT_MAX_DIFFS)
    ap.add_argument("--out", default=str(OUT), help="record to write (merged per artefact)")
    ap.add_argument("--list", action="store_true", help="print the artefact list and exit")
    ap.add_argument("--stamp", metavar="RECORD", nargs="+",
                    help="write the `reproduced` stamp from the RECORDs (later supersedes earlier) "
                         "into the manifests, then exit")
    ap.add_argument("--portable", metavar="RECORD", nargs="+",
                    help="rewrite records without machine paths, then exit")
    ap.add_argument("--allow-main-checkout", action="store_true",
                    help="permit --run outside a linked worktree (see the module docstring)")
    args = ap.parse_args()
    out_path = Path(args.out)
    if args.run and not args.allow_main_checkout and not is_linked_worktree():
        raise SystemExit("--run restores every tracked file that changes while a producer runs and "
                         "lets producers write shared caches; run it in a linked worktree "
                         "(git worktree add) or pass --allow-main-checkout")

    if args.portable:
        for r in args.portable:
            write_record(Path(r), load_record(Path(r)), args.python)
        print(f"  rewrote {len(args.portable)} record(s) without machine paths")
        return 0

    if args.stamp:
        done = stamp([Path(r) for r in args.stamp])
        print(f"  stamped {len(done)} manifest(s): {', '.join(done)}")
        return 0

    if args.list:
        for s in SPECS:
            print(f"{s.stem:40s} {s.producer:24s} {' '.join(s.args)}")
        return 0

    by_stem = {s.stem: s for s in SPECS}
    if args.only:
        unknown = [o for o in args.only if o not in by_stem]
        if unknown:
            raise SystemExit(f"unknown artefact(s) {unknown}; see --list")
        targets = [by_stem[o] for o in args.only]
    else:
        targets = SPECS

    doc = load_record(out_path)
    doc.setdefault("artefacts", {})
    doc.update({
        "purpose": ("does each committed producer, re-run with its recorded parameters, reproduce "
                    "its committed artefact? Measured, per artefact, against git HEAD."),
        "producer_of_this_record": "papers/lure-broadcast/scripts/verify_rerun.py",
        "volatile_rule": ("a field is ignored only if it is wall-clock time, a timestamp or date, a "
                          "duration, a throughput or a git commit; applied as an exact allowlist on "
                          "the last dict key of each difference's path"),
        "volatile_allowlist": sorted(VOLATILE_KEYS),
        "verdicts": ["identical", "identical_except_volatile", "differs", "failed", "timeout"],
        "max_diffs_listed_per_artefact": args.max_diffs,
    })
    for spec in targets:
        print(f"== {spec.stem}: {spec.producer} {' '.join(spec.args)}", flush=True)
        rec = verify_one(spec, args.run, args.python, Path(args.log_dir), args.timeout,
                         args.max_diffs)
        doc["artefacts"][spec.stem] = rec
        order = [s.stem for s in SPECS]
        doc["artefacts"] = dict(sorted(doc["artefacts"].items(),
                                       key=lambda kv: order.index(kv[0]) if kv[0] in order else 999))
        doc["summary"] = {k: v.get("verdict") for k, v in doc["artefacts"].items()}
        write_record(out_path, doc, args.python)
        extra = f"  ({rec.get('differences_total', 0)} differences)" if rec.get("verdict") == "differs" else ""
        print(f"   -> {rec.get('verdict')}  run {rec.get('run_wall_seconds', '-')} s{extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
