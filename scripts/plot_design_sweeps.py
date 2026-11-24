"""Figures for the design sweeps (outputs/design_sweeps/*.csv -> fig_*.png).

Encoding: tilt is ordinal -> one blue ramp light->dark; trackers / east-west racks take the next
categorical hue (orange). Hollow markers = outside the practical envelope. Current design = ring.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vintage import ROOT

D = ROOT / "outputs/design_sweeps"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = {3: ["#86b6ef", "#2a78d6", "#104281"],
        5: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        6: ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281"]}
ORANGE, AQUA = "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})


def series(ax, df, x, y, color, label, marker="o"):
    df = df.sort_values(x)
    ax.plot(df[x], df[y], color=color, label=label, zorder=2)
    f, nf = df[df.feasible], df[~df.feasible]
    ax.scatter(f[x], f[y], s=36, color=color, edgecolor=SURF, linewidth=1.5, marker=marker, zorder=3)
    ax.scatter(nf[x], nf[y], s=36, facecolor=SURF, edgecolor=color, linewidth=1.5, marker=marker, zorder=3)


def mark_current(ax, x, y, text, dx=12, dy=-18):
    ax.scatter([x], [y], s=170, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=4)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=9, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9))


def note(fig, extra="1 MWdc, DC/AC 1.2, unshaded"):
    fig.text(0.01, 0.01, "Hollow markers: outside the practical envelope (row gap, tilt, GCR, DC/AC limits). "
             f"PVWatts v8, College Station TMY, {extra}.", color=INK2, fontsize=8)


def fig_roof():
    r = pd.read_csv(D / "sweep_roof.csv")
    rows = r[(r.layout == "edge-aligned 140") & r.tilt.isin([5, 10, 15])]
    ew = r[r.layout == "east-west dual tilt"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, y, lab in [(axs[0], "kwh_per_m2", "kWh per m² of roof per year"),
                       (axs[1], "kwh_per_kwdc", "kWh per kWdc per year (per module)")]:
        for c, t in zip(BLUE[3], [5, 10, 15]):
            series(ax, rows[rows.tilt == t], "gcr", y, c, f"rows {t}°, facing 140°")
        for ls, t in [((0, (4, 2)), 5), ((0, (1, 1.5)), 10)]:
            e = ew[ew.tilt == t].sort_values("gcr")
            ax.plot(e.gcr, e[y], color=ORANGE, linestyle=ls, label=f"east-west {t}°")
            ax.scatter(e.gcr, e[y], s=36, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3)
        cur = rows[(rows.tilt == 10) & (rows.gcr == 0.76)].iloc[0]
        mark_current(ax, 0.76, cur[y], "current design", dx=-95 if y == "kwh_per_kwdc" else 12,
                     dy=-22 if y == "kwh_per_kwdc" else -18)
        ax.set_xlabel("ground coverage ratio (module area / roof area)")
        ax.set_title(lab, loc="left", color=INK, fontsize=11)
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Flat roofs: denser, flatter racks win per m² of roof; steeper rows win per module",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    note(fig); fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(D / "fig_roof_sweep.png", dpi=200)


def fig_ground():
    g = pd.read_csv(D / "sweep_ground.csv")
    fx, tr = g[g.mount == "fixed"], g[g.mount == "tracker"]
    tilts = [15, 25, 35]  # full 10-35 deg set is in sweep_ground.csv; lines overlap within ~1 %
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, y, lab in [(axs[0], "kwh_per_m2", "kWh per m² of land per year"),
                       (axs[1], "kwh_per_kwdc", "kWh per kWdc per year (per module)")]:
        for c, t in zip(BLUE[3], tilts):
            series(ax, fx[fx.tilt == t], "gcr", y, c, f"fixed {t}°")
        series(ax, tr, "gcr", y, ORANGE, "1-axis tracker", marker="D")
        mark_current(ax, 0.5, fx[(fx.tilt == 25) & (fx.gcr == 0.5)][y].iloc[0], "current fixed", dy=-26 if y == "kwh_per_m2" else 22)
        mark_current(ax, 0.35, tr[tr.gcr == 0.35][y].iloc[0], "current tracker", dy=-22 if y == "kwh_per_m2" else 12)
        ax.set_xlabel("ground coverage ratio")
        ax.set_title(lab, loc="left", color=INK, fontsize=11)
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Open land: fixed tilt packs more energy per m², trackers make more per module",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    note(fig); fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(D / "fig_ground_sweep.png", dpi=200)


def fig_dcac():
    d = pd.read_csv(D / "sweep_dcac.csv")
    cols = {"fixed 25 deg": "#2a78d6", "tracker": ORANGE, "roof 10 deg": AQUA}
    names = {"fixed 25 deg": "ground fixed 25°", "tracker": "1-axis tracker", "roof 10 deg": "roof rows 10°"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, y, lab in [(axs[0], "mwh_per_mwac", "MWh per MWac of inverter per year"),
                       (axs[1], "clipping_pct", "inverter clipping loss [%]")]:
        ax.axvspan(1.1, 1.5, color=GRID, alpha=0.5, zorder=0)
        for m, c in cols.items():
            series(ax, d[d["mount"] == m], "dc_ac", y, c, names[m])
        ax.set_xlabel("DC/AC ratio (MWdc per MWac at a fixed AC size)")
        ax.set_title(lab, loc="left", color=INK, fontsize=11)
    axs[1].axhline(3, color=INK2, linewidth=1, linestyle=(0, (4, 2)))
    axs[1].annotate("3 % clipping", (1.0, 3), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
    axs[0].legend(loc="upper left", fontsize=9)
    note(fig, "fixed AC size (DC added per MWac), unshaded; shaded band = 1.1-1.5 practice")
    fig.suptitle("Grid-limited: oversizing DC to 1.4-1.5 adds 15-22 % energy per MWac for 1-3 % clipping",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(D / "fig_dcac_sweep.png", dpi=200)


if __name__ == "__main__":
    fig_roof(); fig_ground(); fig_dcac()
