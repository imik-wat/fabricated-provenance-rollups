"""Generate the paper's figures from results/*.json in greyscale.

Greyscale rendering
-------------------
Every figure must remain legible in monochrome and must not depend on hue, so series are separated
by marker, line style and greyscale level rather than by colour. Legends are placed outside the
plotting area where an interior position would overlap the data.

Each figure is regenerable by one command from committed artefacts (project rule 3, README), and
no figure is drawn from a number that does not already appear in results/.

Inputs   papers/lure-broadcast/results/{forgery_chain_<chain>,depth_campaigns,criterion_denominators}.json
Outputs  papers/lure-broadcast/figures/*.pdf

Failure modes
-------------
- A missing artefact raises an error rather than producing a figure from which one series is
  absent without indication.

Complexity: trivial; reads small JSON documents and writes vector PDFs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "papers" / "lure-broadcast" / "results"
FIG = ROOT / "papers" / "lure-broadcast" / "figures"
CHAINS = [("arbitrum-one", "Arbitrum One"), ("base", "Base"), ("optimism", "Optimism"),
          ("polygon-zkevm", "Polygon zkEVM"), ("zksync-era", "zkSync Era")]
STYLES = ["-o", "--s", "-.^", ":v", "-D"]
GREYS = ["0.0", "0.30", "0.50", "0.65", "0.45"]


def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing {p.relative_to(ROOT)}; run the producing script first rather "
                         f"than drawing a figure with a series missing")
    return json.loads(p.read_text())


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def fig_depth() -> None:
    """Cost of the rank cutoff: contracts rise steeply while new recipients do not."""
    camp = load("depth_campaigns.json")["per_chain"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for i, (slug, label) in enumerate(CHAINS):
        d = load(f"forgery_chain_{slug}.json")["depth_table"]
        xs = [r["cutoff"] or 1200 for r in d]
        a1.plot(xs, [r["contracts"] for r in d], STYLES[i], color=GREYS[i], ms=3.5, lw=1.1,
                label=label)
        a2.plot(xs, [r["distinct_recipients"] for r in d], STYLES[i], color=GREYS[i], ms=3.5,
                lw=1.1, label=label)
    for ax, ylab in ((a1, "Flagged contracts (cumulative)"),
                     (a2, "Named recipients (cumulative)")):
        ax.set_xlabel("Rank cutoff")
        ax.set_ylabel(ylab)
        ax.set_yscale("symlog")
        ax.set_xticks([100, 300, 600, 1200])
        _style(ax)
    # The legend is placed beneath both panels rather than inside either. Within the axes it
    # overlapped the curves in the upper left of the left panel, which is where three of the five
    # series run, and no interior position is free in both panels at once.
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=len(CHAINS),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.3, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "depth_curve.pdf")
    plt.close(fig)
    print("  figures/depth_curve.pdf")


def fig_fanout() -> None:
    """Fan-out per chain against the broadcast limit of one."""
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    vals, labs = [], []
    for slug, label in CHAINS:
        vals.append(load(f"forgery_chain_{slug}.json")["channel_census"]["fanout_ratio"])
        labs.append(label)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ax.barh([labs[i] for i in order], [vals[i] for i in order],
            color="0.72", edgecolor="0.15", height=0.58, lw=0.7)
    ax.axvline(1.0, color="0.0", lw=1.0, ls="--")
    # Headroom above the longest bar so the reference line can be annotated inside the axes. The
    # annotation previously sat below the shortest bar, where it collided with both that bar and
    # the axis line.
    top = len(vals) - 1
    ax.set_ylim(-0.62, top + 0.95)
    ax.text(1.06, top + 0.55, r"broadcast limit, $\varphi = 1$", fontsize=6.5,
            va="center", ha="left")
    for y, i in enumerate(order):
        ax.text(vals[i] + 0.05, y, f"{vals[i]:.2f}", va="center", fontsize=7)
    # The unit belongs in the caption. Spelled out on the axis it exceeded the column width and
    # was truncated in the rendered figure.
    ax.set_xlabel(r"Fan-out ratio $\varphi$")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.xaxis.grid(True, lw=0.4, color="0.86")
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fanout.pdf")
    plt.close(fig)
    print("  figures/fanout.pdf")


def fig_criterion() -> None:
    """The fabricated-sender criterion against its four denominators."""
    cd = load("criterion_denominators.json")["per_chain"]
    keys = [("contracts", "Contracts"), ("campaigns", "Campaigns"),
            ("transfers", "Volume"), ("recipients", "Recipients")]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    n, w = len(keys), 0.19
    for j, (k, lab) in enumerate(keys):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(CHAINS))]
        ys = [100 * (cd[s][k]["share"] or 0.0) for s, _ in CHAINS]
        ax.bar(xs, ys, width=w, label=lab, color=str(0.25 + 0.18 * j), edgecolor="0.1", lw=0.5)
    ax.set_xticks(range(len(CHAINS)))
    ax.set_xticklabels([l for _, l in CHAINS], fontsize=8)
    ax.set_ylabel("Share satisfying the criterion (\\%)")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center")
    ax.set_ylim(0, 100)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "criterion_denominators.pdf")
    plt.close(fig)
    print("  figures/criterion_denominators.pdf")


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "savefig.bbox": "tight"})
    fig_depth()
    fig_fanout()
    fig_criterion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
