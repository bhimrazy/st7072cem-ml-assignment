"""Shared figure style for the paper: IEEE widths, serif 8 pt text, colour-blind-safe palettes,
seaborn whitegrid styling, vector PDF plus 300 dpi PNG output, and a choropleth helper that draws
the OCHA COD-AB district and province outlines.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import config as C

COL_W, PAGE_W = 3.5, 7.16  # inches: IEEE single column and full page width

# Muted, colour-blind-safe palettes. Hazards and models use Okabe-Ito hues desaturated to about
# 65% (seaborn.desaturate), so each keeps its identity but reads softly; regions use Paul Tol's
# "light" qualitative set; sequential maps use YlGnBu and OrRd at moderate range.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MUTE = 0.65
OKABE_MUTED = [sns.desaturate(c, MUTE) if c != "#000000" else "#333333" for c in OKABE]
HAZARD_COL = {
    "flood": sns.desaturate("#0072B2", MUTE),
    "landslide": sns.desaturate("#D55E00", MUTE),
    "heavy_rain": sns.desaturate("#009E73", MUTE),
    "lightning": sns.desaturate("#CC79A7", MUTE),
}
HAZARD_NAME = {
    "flood": "Flood",
    "landslide": "Landslide",
    "heavy_rain": "Heavy rainfall",
    "lightning": "Lightning",
}
MODEL_COL = {
    "climatology": "#8C8C8C",
    "persistence": "#BDBDBD",
    "lr": sns.desaturate("#0072B2", MUTE),
    "rf": sns.desaturate("#009E73", MUTE),
    "hgb": sns.desaturate("#D55E00", MUTE),
    "svm_linear": sns.desaturate("#CC79A7", MUTE),
    "svm_rbf": sns.desaturate("#E69F00", MUTE),
    "refit_threshold": sns.desaturate("#56B4E9", MUTE),
}
MODEL_NAME = {
    "climatology": "Climatology",
    "persistence": "Persistence",
    "lr": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Gradient boosting",
    "svm_linear": "SVM (linear)",
    "svm_rbf": "SVM (RBF)",
    "refit_threshold": "Rain threshold",
}
# Paul Tol "light" qualitative scheme, ordered so neighbouring regions on the map differ in lightness.
REGION_COL = [
    "#77AADD",
    "#EE8866",
    "#EEDD88",
    "#44BB99",
    "#FFAABB",
    "#BBCC33",
    "#99DDFF",
    "#AAAA00",
]
BELT_COL = {"Terai": "#EEDD88", "Hill": "#44BB99", "Mountain": "#77AADD"}


def shade(color, factor=0.55):
    """A darker tone of a colour, for a second line of the same model."""
    from matplotlib.colors import to_hex, to_rgb

    return to_hex([c * factor for c in to_rgb(color)])


def _soft(name, top=0.72, bottom=0.05):
    """A sequential colormap cut before its darkest end, so maps stay soft in print."""
    from matplotlib.colors import LinearSegmentedColormap

    base = matplotlib.colormaps[name]
    return LinearSegmentedColormap.from_list(f"{name}_soft", base(np.linspace(bottom, top, 256)))


# Sequential colormaps for rates, probabilities and counts.
SEQ_RATE = SEQ_PROB = _soft("YlGnBu")
SEQ_COUNT = _soft("OrRd", 0.78)
OUTLINE_DISTRICT, OUTLINE_PROVINCE, OUTLINE_NATION = "#9A9A9A", "#6F6F6F", "#6F6F6F"
HIT_OUTLINE = sns.desaturate("#D55E00", 0.8)


def setup():
    """Apply the paper's style once per notebook."""
    sns.set_theme(
        style="whitegrid", font="serif", rc={"grid.linewidth": 0.4, "grid.color": "#E3E3E3"}
    )
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.prop_cycle": matplotlib.cycler(color=OKABE_MUTED),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "axes.grid.axis": "y",
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def fig(width=COL_W, height=2.4, **kw):
    """A figure of the given size in inches (single IEEE column by default) and its axes."""
    return plt.subplots(figsize=(width, height), **kw)


def save(fig, name) -> None:
    """Save as vector PDF and 300 dpi PNG under paper/figures/, then show it inline."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(C.FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.show()  # in a notebook this also closes the figure


def letter(ax, s, x=-0.12, y=1.04):
    """Panel letter such as (a) at the top left of an axis."""
    ax.text(
        x,
        y,
        f"({s})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# ---------- maps ----------


def boundaries():
    """District (ADM2) and province (ADM1) polygons from the simplified OCHA COD-AB files in data/raw/."""
    import geopandas as gpd

    d = gpd.read_file(C.RAW / "npl-districts-adm2.geojson").set_index("district")
    p = gpd.read_file(C.RAW / "npl-provinces-adm1.geojson")
    return d, p


def map_axes(ax, districts_gdf, provinces_gdf):
    """Grey basemap fill with thin district borders and darker province borders; no axes."""
    districts_gdf.plot(ax=ax, color="#F4F4F4", edgecolor=OUTLINE_DISTRICT, linewidth=0.3)
    provinces_gdf.boundary.plot(ax=ax, color=OUTLINE_PROVINCE, linewidth=0.6)
    provinces_gdf.dissolve().boundary.plot(ax=ax, color=OUTLINE_NATION, linewidth=0.8)
    ax.set_axis_off()
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.set_xlim(80.0, 88.25)
    ax.set_ylim(26.3, 30.5)
    return ax


def choropleth(
    ax, values, label, cmap=SEQ_RATE, vmin=None, vmax=None, cbar=True, title=None, norm=None
):
    """Colour each district by `values` (a Series indexed by district name) over the basemap.

    The colour bar is a slim horizontal bar below the map with its label underneath, so nothing
    sits on the polygons; `title` goes above the panel (use it for the panel letter). Values above
    `vmax` are clipped and the bar gets an arrow. `norm` (for example PowerNorm) overrides vmin/vmax.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    d, p = boundaries()
    map_axes(ax, d, p)
    g = d.join(values.rename("v"))
    vmin = g["v"].min() if vmin is None else vmin
    vmax = g["v"].max() if vmax is None else vmax
    norm = Normalize(vmin, vmax) if norm is None else norm
    g.plot(
        ax=ax,
        column="v",
        cmap=cmap,
        norm=norm,
        edgecolor=OUTLINE_DISTRICT,
        linewidth=0.3,
        missing_kwds={"color": "#F4F4F4"},
    )
    p.boundary.plot(ax=ax, color=OUTLINE_PROVINCE, linewidth=0.6)
    if title:
        ax.set_title(title, fontsize=8, loc="left", pad=3)
    if cbar:
        cax = ax.inset_axes([0.2, -0.1, 0.6, 0.05])
        clipped = float(values.max()) > float(norm.vmax)
        cb = plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            extend="max" if clipped else "neither",
        )
        cb.set_label(label, fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        cb.outline.set_linewidth(0.4)
    return g


def highlight(ax, names, label="District with a damaging incident", color="#404040"):
    """A small dark grey dot at each named district's representative point, with a legend entry."""
    d, _ = boundaries()
    pts = d.loc[list(names)].representative_point()
    ax.scatter(pts.x, pts.y, s=7, color=color, alpha=0.85, linewidths=0, zorder=5, label=label)
    ax.legend(
        loc="upper right",
        fontsize=6,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.2,
        scatterpoints=1,
        markerscale=1.2,
    )


def categorical_map(ax, labels, colours, names=None, legend_loc="upper right", title=None):
    """Colour each district by a categorical label (Series indexed by district) with a legend."""
    from matplotlib.patches import Patch

    d, p = boundaries()
    map_axes(ax, d, p)
    g = d.join(labels.rename("lab"))
    for k in sorted(g["lab"].dropna().unique(), key=str):
        g[g["lab"] == k].plot(ax=ax, color=colours[k], edgecolor=OUTLINE_DISTRICT, linewidth=0.3)
    p.boundary.plot(ax=ax, color=OUTLINE_PROVINCE, linewidth=0.6)
    if title:
        ax.set_title(title, fontsize=8, loc="left", pad=3)
    handles = [
        Patch(facecolor=colours[k], edgecolor=OUTLINE_DISTRICT, label=(names or {}).get(k, k))
        for k in sorted(g["lab"].dropna().unique(), key=str)
    ]
    ax.legend(
        handles=handles,
        loc=legend_loc,
        fontsize=6,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        handlelength=1.2,
        borderpad=0.4,
    )
    return g


# ---------- expanding-window (rolling-origin) figure ----------


def rolling_pr_reliability_budget(start=2017, best="rf"):
    """Fig. of the main test result from the rolling-origin predictions for 2024 and 2025.

    Same three panels as notebook 05's fixed-split figure: (a) precision-recall curves, (b)
    reliability of the out-of-fold calibrated probabilities, (c) POD against the alert budget.
    Each season has its own fit, so (a) averages the per-season curves (legend: mean per-season
    PR-AUC) and (c) sets the budget within each season, as `E.seasonal_metrics` does; (b) pools the
    calibrated probabilities. Also writes results/rolling-origin-reliability.csv.
    """
    import pandas as pd
    from sklearn.metrics import average_precision_score, precision_recall_curve

    from . import evaluate as E

    pred = pd.read_parquet(C.PROCESSED / "rolling-origin-predictions.parquet")
    te = pred[(pred["start"] == start) & pred["season"].isin([2024, 2025])]
    y, clim, season = te["y"].to_numpy(), te["clim"].to_numpy(), te["season"].to_numpy()
    recall = np.linspace(
        0.005, 1, 200
    )  # from 0.005: precision at recall 0 is a single-row artefact

    def mean_pr(s):
        """Precision on a recall grid averaged over seasons, and the mean per-season PR-AUC."""
        curves, aps = [], []
        for x in np.unique(season):
            i = season == x
            p, r, _ = precision_recall_curve(y[i], s[i])
            curves.append(np.interp(recall, r[::-1], p[::-1]))
            aps.append(average_precision_score(y[i], s[i]))
        return np.mean(curves, axis=0), np.mean(aps)

    def seasonal_pod(s, b):
        alert = np.zeros(len(s), dtype=bool)
        for x in np.unique(season):
            i = np.flatnonzero(season == x)
            alert[i] = E.budget_alerts(s[i], b)
        return E.alert_metrics(y, s, alert=alert)["pod"]

    models = ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]
    legend = {
        "fontsize": 6.5,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.3),
        "columnspacing": 1.0,
        "handlelength": 1.6,
    }
    f, axes = fig(width=COL_W, height=7.6, nrows=3)
    f.subplots_adjust(hspace=0.95)

    ax = axes[0]
    for m in ["climatology"] + models:
        s = clim if m == "climatology" else te[f"score_{m}"].to_numpy()
        p, ap = mean_pr(s)
        ax.plot(
            recall,
            p,
            color=MODEL_COL[m],
            lw=1.5 if m == "lr" else 1.0,
            label=f"{MODEL_NAME[m]} ({ap:.3f})",
        )
    p, ap = mean_pr(te["score_lr_B"].to_numpy())
    ax.plot(
        recall,
        p,
        color=MODEL_COL["lr"],
        ls="--",
        lw=1.0,
        label=f"{MODEL_NAME['lr']}, set B ({ap:.3f})",
    )
    ax.axhline(y.mean(), color="k", lw=0.6, ls=":", label=f"Base rate ({y.mean():.3f})")
    ax.set(xlabel="Recall (POD)", ylabel="Precision", xlim=(0, 1), ylim=(0, 0.5))
    ax.legend(ncol=3, **legend)
    letter(ax, "a")

    ax = axes[1]
    rows = []
    for lab, col, color, ls, mk, mfc in (
        (MODEL_NAME["lr"], "prob_lr", MODEL_COL["lr"], "-", "o", None),
        (MODEL_NAME[best], f"prob_{best}", shade(MODEL_COL[best]), "--", "D", "white"),
        ("Climatology", "clim", MODEL_COL["climatology"], "-", "s", None),
    ):
        t = E.reliability_table(y, te[col].to_numpy(), bins=10)
        ax.plot(
            t["predicted"], t["observed"], marker=mk, ms=3, mfc=mfc, color=color, ls=ls, label=lab
        )
        rows.append(t.assign(model=col.removeprefix("prob_")))
    lim = 0.22
    ax.plot([0, lim], [0, lim], color="k", lw=0.6, ls=":")
    ax.set(
        xlim=(0, lim), ylim=(0, lim), xlabel="Predicted probability", ylabel="Observed frequency"
    )
    ax.legend(ncol=3, **legend)
    letter(ax, "b")

    ax = axes[2]
    budgets = np.linspace(0.01, 0.30, 30)
    for m, col, ls in (
        ("climatology", "clim", "-"),
        (best, f"score_{best}", "-"),
        ("lr", "score_lr", "-"),
        ("lr", "score_lr_B", "--"),
    ):
        s = te[col].to_numpy()
        lab = MODEL_NAME[m] + (", set B" if col.endswith("_B") else "")
        pod = [seasonal_pod(s, b) for b in budgets]
        ax.plot(100 * budgets, pod, color=MODEL_COL[m], ls=ls, label=lab)
    ax.axvline(100 * C.ALERT_BUDGET, color="k", lw=0.6, ls=":")
    ax.set(xlabel="Alert budget (% of district-days)", ylabel="Probability of detection")
    ax.legend(ncol=2, **legend)
    letter(ax, "c")
    save(f, "fig-rolling-pr-reliability-budget")

    rel = pd.concat(rows)
    rel.round(4).to_csv(C.RESULTS / "rolling-origin-reliability.csv", index=False)
    mean = {m: te[f"prob_{m}"].mean() for m in models + ["lr_B"]}
    print("mean predicted probability:", {k: round(float(v), 4) for k, v in mean.items()})
    print(f"observed {y.mean():.4f}, climatology {clim.mean():.4f}")


if __name__ == "__main__":
    import warnings

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
    setup()
    rolling_pr_reliability_budget()
