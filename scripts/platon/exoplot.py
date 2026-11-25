"""exoplot -- house style and plot helpers for exoplanet-science figures.

Distilled from recent figures by the Cowan / Dang (McGill), Artigau / Bouchy
(NIRPS, SPIRou), Helled (Zurich), Queloz (ETH / Cambridge, SPECULOOS),
Komacek (Maryland) and Leconte (Bordeaux) groups. See ../references/ for the
reasoning behind each choice.

Copy this file next to the figure script that uses it (do not import it from
~/.claude) so the figure stays reproducible with the project.

Depends only on numpy + matplotlib (>= 3.6).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter

# --------------------------------------------------------------------------
# Page geometry (inches). Design every figure at its printed size.
# --------------------------------------------------------------------------
JOURNAL_WIDTHS = {
    # journal: (single column, full page)
    "aas": (3.50, 7.10),      # ApJ / AJ / PSJ / ApJL (AASTeX two-column)
    "aanda": (3.46, 7.09),    # A&A: 88 mm / 180 mm (120 mm = 4.72 in also allowed)
    "mnras": (3.31, 6.97),    # MNRAS: 84 mm / 177 mm
    "nature": (3.50, 7.20),   # Nature / Nature Astronomy: 89 mm / 183 mm
}

# Labels: AAS and MNRAS put units in parentheses, A&A in square brackets.
UNIT_BRACKETS = {"aas": "()", "mnras": "()", "nature": "()", "aanda": "[]"}

# --------------------------------------------------------------------------
# Colours. Fixed meanings: never reuse a molecule or instrument colour for
# something else within one paper.
# --------------------------------------------------------------------------
MOLECULE_COLORS = {
    "H2O": "#1f77b4", "CO2": "#d62728", "CH4": "#9467bd", "CO": "#2ca02c",
    "SO2": "#ff7f0e", "NH3": "#e377c2", "H2S": "#bcbd22", "HCN": "#17becf",
    "SiO": "#8c564b", "TiO": "#7f7f7f", "VO": "#525252", "FeH": "#a65628",
    "Na": "#f0c808", "K": "#b8860b", "He": "#00a6a6", "H-": "#404040",
    "OCS": "#6b8e23", "C2H2": "#8dd3c7",
}
MOLECULE_TEX = {
    "H2O": r"H$_2$O", "CO2": r"CO$_2$", "CH4": r"CH$_4$", "SO2": r"SO$_2$",
    "NH3": r"NH$_3$", "H2S": r"H$_2$S", "C2H2": r"C$_2$H$_2$", "H-": r"H$^-$",
}
INSTRUMENT_COLORS = {
    # JWST
    "NIRISS SOSS": "#1f4e9c", "NIRSpec PRISM": "#6a3d9a", "NRS1": "#2c7bb6",
    "NRS2": "#d7191c", "G395H": "#c0392b", "NIRCam": "#1a9850", "MIRI LRS": "#8c510a",
    "MIRI": "#8c510a",
    # others
    "HST WFC3": "#ff7f0e", "Spitzer": "#e31a1c", "TESS": "#1f78b4", "CHEOPS": "#33a02c",
    # RV spectrographs (NIRPS dark red / HARPS orange as in the NIRPS GTO papers)
    "NIRPS": "#8b0000", "HARPS": "#f39c12", "ESPRESSO": "#008080", "SPIRou": "#1b7837",
    "CARMENES": "#3366cc", "HARPS-N": "#9b59b6", "MAROON-X": "#2ca02c",
}
INSTRUMENT_MARKERS = {"NIRPS": "o", "HARPS": "s", "ESPRESSO": "D", "SPIRou": "^",
                      "CARMENES": "v", "HARPS-N": "P", "MAROON-X": "*"}
DAYSIDE = "#e66101"      # orange: dayside / emission / hot
NIGHTSIDE = "#2c3e91"    # deep blue: nightside / cold
REFLECTED = "#2166ac"
MODEL_GREY = "0.35"
RAW_GREY = "0.6"

# Interior layer colours (Helled-group wedge diagrams).
LAYER_COLORS = {"H-He": "#fdbf6f", "H2O": "#1f78b4", "MgSiO3": "#8c564b",
                "Mg2SiO4": "#8c564b", "rock": "#8c564b", "Fe": "#c7c7c7"}

# Colormaps by meaning (all perceptually ordered except the wavelength one).
CMAPS = {
    "temperature": "inferno",      # T maps, brightness maps, eclipse maps
    "teq": "RdYlBu_r",             # planets coloured by Teq / insolation
    "wavelength": "exo_wavelength",  # spectroscopic stacks: turbo without its dark ends
    "phase": "twilight_shifted",   # orbital phase (cyclic)
    "diverging": "RdBu_r",         # winds, vertical velocity, residual maps
    "cloud": "Purples",            # cloud column / condensate maps
    "density": "viridis",          # anything else sequential
    "teq_stack": "exo_teq_stack",  # population stacks: blue (cool) -> orange (hot), no pale middle
}
if "exo_wavelength" not in mpl.colormaps:
    mpl.colormaps.register(LinearSegmentedColormap.from_list(
        "exo_wavelength", mpl.colormaps["turbo"](np.linspace(0.08, 0.92, 256))))
if "exo_teq_stack" not in mpl.colormaps:
    mpl.colormaps.register(LinearSegmentedColormap.from_list(
        "exo_teq_stack", ["#1f3b8f", "#7b5ea7", "#e66101"]))


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def style_rc(fontsize: float = 8.0, family: str = "sans", usetex: bool = False) -> dict:
    """rcParams for the house style: closed box, inward ticks on all sides."""
    rc = {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "legend.fontsize": fontsize - 1,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "axes.linewidth": 0.7,
        "axes.ymargin": 0.08,     # room for panel tags / labels above the data
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.labelpad": 2.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "errorbar.capsize": 0,
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-4, 5),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable TrueType text in PDF/EPS
        "svg.fonttype": "none",
        "image.cmap": "viridis",
        "axes.prop_cycle": mpl.cycler(color=["k", "#2c7bb6", "#d7191c", "#1a9850",
                                             "#ff7f0e", "#6a3d9a", "#8c564b"]),
    }
    if usetex:
        rc.update({"text.usetex": True, "font.family": "serif",
                   "text.latex.preamble": r"\usepackage{amsmath}\usepackage{txfonts}"})
    elif family == "serif":
        rc.update({"font.family": "serif",
                   "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                   "mathtext.fontset": "stix"})
    else:
        rc.update({"font.family": "sans-serif",
                   "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                   "mathtext.fontset": "dejavusans"})
    return rc


def use_style(fontsize: float = 8.0, family: str = "sans", usetex: bool = False) -> None:
    """Apply the house style globally. Call once, before creating figures."""
    mpl.rcParams.update(style_rc(fontsize, family, usetex))


def figsize(journal: str = "aas", span: str = "single", aspect: float = 0.75,
            width: float | None = None) -> tuple[float, float]:
    """(w, h) in inches at printed size. span: 'single' | 'double' | fraction of full width."""
    single, full = JOURNAL_WIDTHS[journal]
    if width is None:
        width = {"single": single, "double": full}.get(span) if isinstance(span, str) else full * span
    return (width, width * aspect)


def label(name: str, unit: str | None = None, journal: str = "aas") -> str:
    """Axis label with the journal's unit brackets: label('Wavelength', r'$\\mu$m')."""
    if not unit:
        return name
    o, c = UNIT_BRACKETS[journal]
    return f"{name} {o}{unit}{c}"


def mol(name: str) -> str:
    """Mathtext species label, e.g. mol('CO2') -> CO$_2$."""
    return MOLECULE_TEX.get(name, name)


def panel_label(ax, text: str, style: str = "nature", loc: str = "upper left",
                inside: bool = True, **kw):
    """Panel tag. style: 'nature' -> bold lowercase 'a'; 'paren' -> '(a)'; 'boxed' -> 'A' in a rounded box."""
    x, y = {"upper left": (0.03, 0.96), "upper right": (0.97, 0.96),
            "lower left": (0.03, 0.04), "lower right": (0.97, 0.04)}[loc]
    if not inside:
        x, y = (-0.02 if "left" in loc else 1.02), 1.02
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    if not inside:
        ha, va = ("right" if "left" in loc else "left"), "bottom"
    props = dict(transform=ax.transAxes, ha=ha, va=va)
    if style == "nature":
        props.update(fontweight="bold", fontsize=mpl.rcParams["font.size"] + 1)
    elif style == "paren":
        text = f"({text})"
    elif style == "boxed":
        props.update(bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=0.6))
    props.update(kw)
    return ax.text(x, y, text, **props)


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def bin_data(x, y, yerr=None, width=None, nbins=None, edges=None):
    """Bin y(x) into bins of `width` (x units), `nbins`, or explicit `edges`.

    Returns (xb, yb, eb). With yerr: inverse-variance mean and its error.
    Without: plain mean and standard error. Empty bins are dropped.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if edges is None:
        if width is not None:
            edges = np.arange(np.nanmin(x), np.nanmax(x) + width, width)
        else:
            edges = np.linspace(np.nanmin(x), np.nanmax(x), (nbins or 30) + 1)
    idx = np.digitize(x, edges) - 1
    xb, yb, eb = [], [], []
    for k in range(len(edges) - 1):
        m = (idx == k) & np.isfinite(y)
        if m.sum() < 1:
            continue
        if yerr is not None:
            w = 1.0 / np.asarray(yerr, float)[m] ** 2
            yb.append(np.sum(w * y[m]) / w.sum()); eb.append(w.sum() ** -0.5)
        else:
            yb.append(y[m].mean())
            eb.append(y[m].std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else np.nan)
        xb.append(x[m].mean())
    return np.array(xb), np.array(yb), np.array(eb)


def constant_r_edges(wmin: float, wmax: float, R: float) -> np.ndarray:
    """Wavelength bin edges at constant resolving power R = lambda / d(lambda)."""
    n = int(np.ceil(np.log(wmax / wmin) / np.log1p(1.0 / R)))
    return wmin * (1.0 + 1.0 / R) ** np.arange(n + 1)


# --------------------------------------------------------------------------
# Generic marks
# --------------------------------------------------------------------------
def raw_and_binned(ax, x, y, yerr=None, bin_width=None, nbins=None, color=RAW_GREY,
                   binned_color="k", raw_alpha=0.3, raw_size=1.5, binned_ms=3.5,
                   label_binned=None, zorder=3):
    """Faint unbinned points (rasterized) under open-circle binned points with error bars."""
    # Raw points never drive the axis limits (the binned points and model do):
    # add_artist, unlike plot/add_line, leaves the data limits untouched.
    raw = mpl.lines.Line2D(x, y, ls="none", marker=".", ms=raw_size, color=color,
                           alpha=raw_alpha, mew=0, zorder=zorder - 2, rasterized=True)
    ax.add_artist(raw)
    xb, yb, eb = bin_data(x, y, yerr, width=bin_width, nbins=nbins)
    ax.errorbar(xb, yb, eb, fmt="o", ms=binned_ms, mfc="white", mec=binned_color,
                ecolor=binned_color, mew=0.7, elinewidth=0.7, zorder=zorder,
                label=label_binned)
    return xb, yb, eb


def data_points(ax, x, y, yerr=None, xerr=None, color="k", open_=True, ms=3.5, label=None,
                zorder=5, **kw):
    """Final data points: open circles, black edges, no caps (the Cowan/NIRPS default)."""
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=kw.pop("fmt", "o"), ms=ms,
                       mfc="white" if open_ else color, mec=color, ecolor=color, mew=0.7,
                       elinewidth=0.7, label=label, zorder=zorder, **kw)


def model_band(ax, x, median, lo1=None, hi1=None, lo2=None, hi2=None, color="#2c7bb6",
               label=None, lw=1.0, alpha1=0.45, alpha2=0.22, zorder=2):
    """Model median with 1 sigma (darker) and 2 sigma (lighter) envelopes of the same hue."""
    if lo2 is not None:
        ax.fill_between(x, lo2, hi2, color=color, alpha=alpha2, lw=0, zorder=zorder,
                        label=None if label is None else rf"{label} 2$\sigma$")
    if lo1 is not None:
        ax.fill_between(x, lo1, hi1, color=color, alpha=alpha1, lw=0, zorder=zorder,
                        label=None if label is None else rf"{label} 1$\sigma$")
    return ax.plot(x, median, color=color, lw=lw, zorder=zorder + 1,
                   label=None if label is None else label)[0]


def zero_line(ax, y=0.0, **kw):
    kw.setdefault("color", "0.4"); kw.setdefault("ls", "--"); kw.setdefault("lw", 0.7)
    kw.setdefault("zorder", 1)
    return ax.axhline(y, **kw)


def upper_limit(ax, x, y, color="k", frac=0.08, ms=6, **kw):
    """Upper limit at y: a bar with a downward arrow `frac` of the y-range long.

    Use for a 2- or 3-sigma limit on a depth / amplitude / albedo; say which in the caption.
    """
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        dy = y - y / 10 ** (frac * np.log10(hi / lo))
    else:
        dy = frac * (hi - lo)
    ax.plot([x], [y], "_", color=color, ms=ms, mew=1.0, **kw)
    ax.annotate("", xy=(x, y - dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=7,
                                shrinkA=0, shrinkB=0))


def log_wavelength_axis(ax, ticks=(0.6, 0.8, 1, 1.5, 2, 3, 4, 5), lim=None):
    """Log-scaled wavelength axis with plain-number major ticks (Piaulet-Ghorayeb style)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if lim:
        ax.set_xlim(*lim)


def shade_bands(ax, bands: Mapping[str, Sequence[tuple[float, float]]], alpha=0.14,
                text=True, y_text=0.97, fontsize=None):
    """Pale vertical spans marking where each species absorbs, labelled at the top.

    bands = {"H2O": [(1.3, 1.5), (1.75, 2.05)], "CO2": [(4.2, 4.45)]}
    """
    for sp, spans in bands.items():
        c = MOLECULE_COLORS.get(sp, "0.5")
        for a, b in spans:
            ax.axvspan(a, b, color=c, alpha=alpha, lw=0, zorder=0)
            if text:
                ax.text(np.sqrt(a * b) if ax.get_xscale() == "log" else 0.5 * (a + b), y_text,
                        mol(sp), color=c, ha="center", va="top", transform=ax.get_xaxis_transform(),
                        fontsize=fontsize or mpl.rcParams["font.size"] - 1)


def colorbar(fig, mappable, ax, label=None, **kw):
    kw.setdefault("pad", 0.02); kw.setdefault("fraction", 0.05)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.ax.tick_params(direction="in", which="both", width=0.5)
    cb.outline.set_linewidth(0.6)
    if label:
        cb.set_label(label)
    return cb


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------
def flux_residual_axes(fig=None, ncols=1, height_ratios=(3, 1), size=None, sharey_row=True):
    """Columns of [flux; residual] panels with shared x, no vertical gap.

    Returns (fig, axes) with axes shape (2, ncols).
    """
    if fig is None:
        fig = plt.figure(figsize=size or figsize(span="single", aspect=0.9))
    gs = fig.add_gridspec(2, ncols, height_ratios=height_ratios, hspace=0.0, wspace=0.06)
    axes = np.empty((2, ncols), dtype=object)
    for j in range(ncols):
        axes[0, j] = fig.add_subplot(gs[0, j], sharey=axes[0, 0] if (sharey_row and j) else None)
        axes[1, j] = fig.add_subplot(gs[1, j], sharex=axes[0, j],
                                     sharey=axes[1, 0] if (sharey_row and j) else None)
        axes[0, j].tick_params(labelbottom=False)
        # keep tick labels from colliding where the two panels meet
        axes[0, j].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="lower"))
        axes[1, j].yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        if j and sharey_row:
            axes[0, j].tick_params(labelleft=False); axes[1, j].tick_params(labelleft=False)
    return fig, axes


def plot_lightcurve(ax_flux, ax_res, t, flux, model_t, model_f, model_at_data, err=None,
                    bin_width=None, color=RAW_GREY, model_color="#d7191c", res_unit="ppm",
                    shade=None, t_label=None, journal="aas"):
    """White-light transit/eclipse: raw + binned flux, model, residuals in ppm.

    shade: list of (t0, t1) spans (e.g. ingress/egress or eclipse duration), drawn pale grey.
    """
    raw_and_binned(ax_flux, t, flux, err, bin_width=bin_width, color=color)
    ax_flux.plot(model_t, model_f, color=model_color, lw=1.1, zorder=4)
    scale = 1e6 if res_unit == "ppm" else 1.0
    res = (np.asarray(flux) - np.asarray(model_at_data)) * scale
    raw_and_binned(ax_res, t, res, None if err is None else np.asarray(err) * scale,
                   bin_width=bin_width, color=color)
    zero_line(ax_res)
    for a in (ax_flux, ax_res):
        for t0, t1 in (shade or []):
            a.axvspan(t0, t1, color="0.9", lw=0, zorder=0)
    ax_flux.set_ylabel("Relative flux")
    ax_res.set_ylabel(label("Residuals", res_unit, journal))
    ax_res.set_xlabel(t_label or label("Time from mid-transit", "h", journal))
    rms = np.nanstd(res)
    ax_res.text(0.02, 0.06, f"rms = {rms:.0f} {res_unit}", transform=ax_res.transAxes,
                fontsize=mpl.rcParams["font.size"] - 1.5, va="bottom")
    return res


def stack_lightcurves(ax, t, fluxes, models=None, wavelengths=None, offset=0.004, bin_width=None,
                      cmap=CMAPS["wavelength"], fmt_label="{:.2f}–{:.2f} µm", label_edges=None,
                      label_x=0.02):
    """Spectroscopic light curves stacked bluest at top, coloured by wavelength.

    fluxes: (n_channel, n_time). models: same shape or None. wavelengths: centres for colour.
    label_edges: list of (lo, hi) to print above each curve.
    """
    n = len(fluxes)
    cm = mpl.colormaps[cmap]
    cols = [cm(v) for v in (np.linspace(0, 1, n) if n > 1 else [0.0])]
    for k in range(n):
        off = -k * offset
        xb, yb, eb = bin_data(t, fluxes[k], width=bin_width, nbins=None if bin_width else 60)
        ax.plot(xb, yb + off, "o", ms=2.2, mfc=cols[k], mec="k", mew=0.25, zorder=3)
        if models is not None:
            ax.plot(t, np.asarray(models[k]) + off, color="k", lw=0.7, zorder=4)
        if label_edges is not None:
            ax.text(label_x, 1.0 + off + 0.12 * offset, fmt_label.format(*label_edges[k]),
                    transform=ax.get_yaxis_transform(), fontsize=mpl.rcParams["font.size"] - 1.5,
                    va="bottom", ha="left")
    ax.set_ylabel("Relative flux + offset")
    return cols


def plot_spectrum(ax, wl, depth, err, wl_halfwidth=None, models: Mapping | None = None,
                  unit="ppm", log_x=True, ticks=None, journal="aas", data_label="Data",
                  ylabel=None):
    """Transmission/emission spectrum: open-circle data over model lines or bands.

    models: {label: dict(wl=..., y=..., color=..., lo1=, hi1=, lo2=, hi2=, ls=)}.
    """
    for name, m in (models or {}).items():
        if "lo1" in m:
            model_band(ax, m["wl"], m["y"], m.get("lo1"), m.get("hi1"), m.get("lo2"), m.get("hi2"),
                       color=m.get("color", "#2c7bb6"), label=name, lw=m.get("lw", 1.0))
        else:
            ax.plot(m["wl"], m["y"], color=m.get("color", MODEL_GREY), lw=m.get("lw", 0.9),
                    ls=m.get("ls", "-"), label=name, zorder=2)
    data_points(ax, wl, depth, err, xerr=wl_halfwidth, label=data_label)
    if log_x:
        log_wavelength_axis(ax, ticks or (0.6, 0.8, 1, 1.5, 2, 3, 4, 5))
    ax.set_xlabel(label("Wavelength", r"$\mu$m", journal))
    ax.set_ylabel(ylabel or label("Transit depth", unit, journal))


def plot_phase_curve(ax, ax_res, phase, fp, err, model_phase, model_fp, model_at_data,
                     components: Mapping | None = None, bin_n=40, unit="ppm", journal="aas",
                     eclipse_phase=0.5, transit_mask=None):
    """Phase curve in Fp/F*: binned open circles, model, optional component curves with bands.

    components: {label: dict(y=..., lo=..., hi=..., color=...)} on model_phase.
    """
    ph, fp = np.asarray(phase), np.asarray(fp)
    keep = np.ones_like(ph, bool) if transit_mask is None else ~transit_mask
    raw_and_binned(ax, ph[keep], fp[keep], None if err is None else np.asarray(err)[keep],
                   nbins=bin_n, raw_alpha=0.15)
    for name, c in (components or {}).items():
        if "lo" in c:
            ax.fill_between(model_phase, c["lo"], c["hi"], color=c["color"], alpha=0.2, lw=0)
        ax.plot(model_phase, c["y"], color=c["color"], lw=1.0, label=name)
    ax.plot(model_phase, model_fp, color="k", lw=1.1, label="Total model", zorder=4)
    for a in (ax, ax_res):
        a.axvline(eclipse_phase, color="0.5", ls=":", lw=0.7, zorder=0)
    raw_and_binned(ax_res, ph[keep], fp[keep] - np.asarray(model_at_data)[keep], nbins=bin_n,
                   raw_alpha=0.15)
    zero_line(ax_res)
    ax.set_ylabel(label(r"$F_{\rm p}/F_\star$", unit, journal))
    ax_res.set_ylabel(label("Residuals", unit, journal))
    ax_res.set_xlabel("Orbital phase")


def plot_tp(ax, T, P_bar, lo1=None, hi1=None, lo2=None, hi2=None, color="k", label_=None,
            journal="aas", ls="-"):
    """T-P profile: log pressure increasing downward, 1/2 sigma envelopes in x."""
    if lo2 is not None:
        ax.fill_betweenx(P_bar, lo2, hi2, color=color, alpha=0.18, lw=0)
    if lo1 is not None:
        ax.fill_betweenx(P_bar, lo1, hi1, color=color, alpha=0.35, lw=0)
    ax.plot(T, P_bar, color=color, ls=ls, lw=1.1, label=label_)
    ax.set_yscale("log")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(label("Temperature", "K", journal))
    ax.set_ylabel(label("Pressure", "bar", journal))


def contribution_bar(ax, x, p_lo, p_hi, color, lw=2.0):
    """Vertical bar marking the pressure range a band probes (beside a T-P profile)."""
    ax.plot([x, x], [p_lo, p_hi], color=color, lw=lw, solid_capstyle="butt")


def plot_posterior(ax, samples, color="#2c7bb6", label_=None, bins=40, filled=True,
                   range_=None, weights=None, density=True, limit=None):
    """1-D posterior as a filled step histogram; `limit`='upper'/'lower' adds a 95% arrow."""
    s = np.asarray(samples)
    h, e = np.histogram(s, bins=bins, range=range_, weights=weights, density=density)
    if filled:
        ax.stairs(h, e, fill=True, color=color, alpha=0.35, lw=0)
    ax.stairs(h, e, color=color, lw=0.9, label=label_)
    ax.set_yticks([]); ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_ylim(0, None)
    if limit:
        q = np.percentile(s, 95 if limit == "upper" else 5)
        ytop = h.max()
        ax.annotate("", xy=(q + (-1 if limit == "upper" else 1) * 0.15 * (e[-1] - e[0]), 0.12 * ytop),
                    xytext=(q, 0.12 * ytop),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=7))
        ax.plot([q, q], [0.04 * ytop, 0.2 * ytop], color=color, lw=1.0)
    return h, e


def plot_rv_phased(ax, ax_res, phase, rv, err, instrument, model_phase, model_rv,
                   resid=None, draws=None, nbins=10, journal="aas"):
    """Phase-folded RVs per instrument + Keplerian, binned points, O-C panel.

    instrument: array of instrument names (keys of INSTRUMENT_COLORS / _MARKERS).
    draws: (n_draw, n_phase) posterior Keplerians, drawn thin grey.
    """
    phase, rv, err, inst = map(np.asarray, (phase, rv, err, instrument))
    if draws is not None:
        for d in draws:
            ax.plot(model_phase, d, color="0.5", lw=0.3, alpha=0.25, zorder=1)
    for name in dict.fromkeys(inst):
        m = inst == name
        kw = dict(fmt=INSTRUMENT_MARKERS.get(name, "o"), ms=2.8, color=INSTRUMENT_COLORS.get(name, "0.3"),
                  alpha=0.6, elinewidth=0.6, mew=0, label=name, zorder=2)
        ax.errorbar(phase[m], rv[m], err[m], **kw)
        if resid is not None:
            ax_res.errorbar(phase[m], np.asarray(resid)[m], err[m], **{**kw, "label": None})
    ax.plot(model_phase, model_rv, color="k", lw=1.1, zorder=3, label="Keplerian")
    xb, yb, eb = bin_data(phase, rv, err, edges=np.linspace(-0.5, 0.5, nbins + 1))
    ax.errorbar(xb, yb, eb, fmt="o", ms=4.5, mfc="white", mec="#08306b", ecolor="#08306b",
                mew=1.0, elinewidth=1.0, zorder=4, label="Binned")
    zero_line(ax)
    if resid is not None:
        xb, yb, eb = bin_data(phase, resid, err, edges=np.linspace(-0.5, 0.5, nbins + 1))
        ax_res.errorbar(xb, yb, eb, fmt="o", ms=4.5, mfc="white", mec="#08306b",
                        ecolor="#08306b", mew=1.0, elinewidth=1.0, zorder=4)
        zero_line(ax_res)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="lower"))
        ax_res.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        ax_res.set_ylabel(label("O $-$ C", r"m s$^{-1}$", journal))
        ax_res.set_xlabel("Orbital phase")
    ax.set_ylabel(label("RV", r"m s$^{-1}$", journal))
    ax.set_xlim(-0.5, 0.5)


def plot_periodogram(ax, period, power, fap: Mapping[str, float] | None = None,
                     mark: Mapping[str, float] | None = None, color="k", journal="aas"):
    """GLS periodogram on a log period axis with FAP levels and marked periods."""
    ax.plot(period, power, color=color, lw=0.7)
    styles = [("--", "#2166ac"), (":", "#1a9850"), ("-.", "#b2182b")]
    for (name, lev), (ls, c) in zip((fap or {}).items(), styles):
        ax.axhline(lev, ls=ls, color=c, lw=0.8, label=f"FAP {name}")
    for name, p in (mark or {}).items():
        ax.axvline(p, color="#d7191c", lw=0.8, alpha=0.8, zorder=0)
        ax.text(p, 0.98, f" {name}", rotation=90, va="top", ha="right", color="#d7191c",
                transform=ax.get_xaxis_transform(), fontsize=mpl.rcParams["font.size"] - 1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(label("Period", "d", journal))
    ax.set_ylabel("Power")
    ax.set_ylim(0, None)


def plot_mass_radius(ax, M, R, Merr=None, Rerr=None, c=None, cmap=CMAPS["teq"], norm=None,
                     curves: Mapping | None = None, highlight: Mapping | None = None,
                     background: tuple | None = None, log=True, journal="aas"):
    """Mass-radius diagram: grey background population, coloured sample, composition curves.

    curves: {label: dict(M=..., R=..., color=..., ls=...)} (load published tables -- Zeng+2016/2019,
            Lopez & Fortney 2014, Aguichine+2021 -- never hand-type approximations).
    highlight: {name: (M, R)} -> black-edged star marker + text label.
    background: (M_all, R_all) archive population drawn small and light grey.
    """
    if background is not None:
        ax.plot(*background, ".", color="0.8", ms=2, mew=0, zorder=0, rasterized=True)
    for name, cv in (curves or {}).items():
        ax.plot(cv["M"], cv["R"], color=cv.get("color", "0.3"), ls=cv.get("ls", "-"), lw=1.0,
                label=name, zorder=1)
    if Merr is not None or Rerr is not None:
        ax.errorbar(M, R, yerr=Rerr, xerr=Merr, fmt="none", ecolor="0.3", elinewidth=0.6, zorder=2)
    sc = ax.scatter(M, R, c=c, cmap=cmap, norm=norm, s=18, edgecolors="k", linewidths=0.5,
                    zorder=3)
    for name, (m, r) in (highlight or {}).items():
        ax.scatter([m], [r], marker="*", s=90, c="gold", edgecolors="k", linewidths=0.7, zorder=5)
        ax.annotate(name, (m, r), xytext=(4, 4), textcoords="offset points",
                    fontsize=mpl.rcParams["font.size"] - 1)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(label("Mass", r"$M_\oplus$", journal))
    ax.set_ylabel(label("Radius", r"$R_\oplus$", journal))
    return sc


def plot_map(ax, lon, lat, field, cmap=CMAPS["temperature"], vmin=None, vmax=None,
             terminators=True, substellar=True, contours=None, journal="aas"):
    """Equirectangular lon-lat map (GCM output, eclipse map); dashed terminators at +/-90 deg."""
    pm = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                       rasterized=True)
    if contours is not None:
        cs = ax.contour(lon, lat, field, levels=contours, colors="k", linewidths=0.5)
        ax.clabel(cs, fontsize=mpl.rcParams["font.size"] - 2, fmt="%g")
    if terminators:
        for x in (-90, 90):
            ax.axvline(x, color="w", ls="--", lw=0.8)
    if substellar:
        ax.plot(0, 0, "+", color="w", ms=6, mew=1.0)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel(label("Longitude", r"$^\circ$", journal))
    ax.set_ylabel(label("Latitude", r"$^\circ$", journal))
    return pm


def offset_stack(ax, rows: Sequence[Mapping], offset: float, cmap=CMAPS["teq_stack"],
                 by_rank=True, norm=None, label_x=None, fmt=None):
    """Population spectra stacked with constant offsets (Sing+2016 style).

    rows: [dict(wl, y, yerr, name, value)] in plotting order, first row at the TOP
    (sort coolest first). Colour follows the RANK of `value` by default, so a sample
    bunched in Teq still spans the full colour range. Points only -- never joined by lines.
    """
    cm = mpl.colormaps[cmap]
    vals = np.array([r.get("value", k) for k, r in enumerate(rows)], float)
    if by_rank:
        vals = np.argsort(np.argsort(vals)).astype(float)
    norm = norm or mpl.colors.Normalize(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() + 1)
    for k, r in enumerate(rows):
        off = (len(rows) - 1 - k) * offset
        col = cm(norm(vals[k]))
        ax.errorbar(r["wl"], np.asarray(r["y"]) + off, r.get("yerr"), fmt="o", ms=2.6,
                    mfc=col, mec="k", mew=0.3, ecolor=col, elinewidth=0.7)
        if label_x is not None:
            ax.text(label_x, np.nanmedian(r["y"]) + off, r["name"] if fmt is None else fmt(r),
                    va="center", ha="left", fontsize=mpl.rcParams["font.size"] - 1, color=col)
    return norm


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def finalize(fig, out_stem, formats=("pdf", "png", "svg"), dpi=300, close=True):
    """Save fig to every format (PDF + PNG + SVG by default) and return the paths."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    fig.align_ylabels()
    for ext in formats:
        p = out_stem.with_suffix("." + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths
