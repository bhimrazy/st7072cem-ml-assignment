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
