"""
Utils file for focal series reconstruction algorithms.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_device(prefer: str = "auto") -> torch.device:
    """Get device in order to do calculations on GPU if possible."""
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_numpy(a):
    """Convert a NumPy / PyTorch / CuPy array to a NumPy array (moves GPU data to CPU)."""
    if hasattr(a, "detach"):      # torch tensor
        a = a.detach().cpu().numpy()
    elif hasattr(a, "get"):       # cupy array
        a = a.get()
    return np.asarray(a)

def normalize(img, minmax):
    """Normalize an image to the given range."""
    img = img.astype("float64")
    img = img - img.min()
    rng = img.max()
    if rng > 0:
        img = img / rng
    img = img * (minmax[1] - minmax[0]) + minmax[0]
    return img
 
def plot_1d(y, x=None, label=None, xlabel="", ylabel="", title="",
            ax=None, save=None, **kwargs):
    """Plot a 1D curve, e.g. a CTF versus spatial frequency.
 
    y      : 1D data
    x      : optional x values (defaults to 0, 1, 2, ...)
    ax     : pass an existing axes to overlay several curves
    save   : filename to save the figure (e.g. "ctf.png")
    kwargs : forwarded to ax.plot (color, linestyle, ...)
    """
    y = to_numpy(y)
    x = np.arange(len(y)) if x is None else to_numpy(x)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
 
    ax.plot(x, y, label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    ax.grid(alpha=0.3)
 
    if save:
        ax.figure.savefig(save, dpi=300, bbox_inches="tight")
    return ax
 
 
def plot_2d(img, mode="abs", x=None, y=None, extent=None, cmap="gray",
            log=False, vmin=None, vmax=None, colorbar=True,
            xlabel="", ylabel="", title="", ax=None, save=None):
    """Plot a 2D array, e.g. an image or a Fourier transform.

    img   : 2D data, real or complex
    mode  : for complex data, what to show: "abs", "phase", "real" or "imag"
    x, y  : optional 1D coordinate arrays for the columns (x) and rows (y),
            e.g. spec.x, spec.y in nm or a centered k axis in 1/nm.
            Must be evenly spaced, since imshow draws uniform pixels.
    extent: (left, right, bottom, top), alternative to x/y; not both
    log   : show log10(1 + value), useful for diffraction / Fourier data
    """
    img = to_numpy(img)

    if (x is not None or y is not None) and extent is not None:
        raise ValueError("Give either extent or x/y, not both.")

    if x is not None or y is not None:
        ny, nx = img.shape[-2:]
        x = np.arange(nx) if x is None else to_numpy(x)
        y = np.arange(ny) if y is None else to_numpy(y)
        if len(x) != nx or len(y) != ny:
            raise ValueError(
                f"Axis lengths (y={len(y)}, x={len(x)}) do not match "
                f"image shape {img.shape}.")
        dx = (x[-1] - x[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (y[-1] - y[0]) / (ny - 1) if ny > 1 else 1.0
        if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)):
            raise ValueError("x and y must be evenly spaced.")
        # half-pixel padding so pixel centers sit on the given coordinates;
        # y runs top to bottom because origin="upper"
        extent = (x[0] - dx / 2, x[-1] + dx / 2,
                  y[-1] + dy / 2, y[0] - dy / 2)

    if np.iscomplexobj(img):
        img = {"abs": np.abs, "phase": np.angle,
               "real": np.real, "imag": np.imag}[mode](img)
        if mode == "phase" and cmap == "gray":
            cmap = "twilight"     # cyclic colormap suits phase data

    if log:
        img = np.log10(1 + np.abs(img))

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(img, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax,
                   origin="upper")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save:
        ax.figure.savefig(save, dpi=300, bbox_inches="tight")
    return ax


def plot_focal_series(focal_series, defoci=None, ncols=None, step=1,
                      shared_scale=True, figsize_per_panel=2.5,
                      suptitle="", save=None, **plot_kwargs):
    """Plot a focal series as a grid of images.

    focal_series : array of shape (N, H, W), defocus along axis 0
    defoci       : optional length-N array, used for the panel titles
    ncols        : columns in the grid (default: roughly square layout)
    step         : show every `step`-th image, useful for long series
    shared_scale : use one vmin/vmax for all panels so contrast is
                   comparable between defoci
    **plot_kwargs: passed to plot_2d (cmap, x, y, extent, log, ...)
    """
    focal_series = to_numpy(focal_series)
    if focal_series.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got shape {focal_series.shape}.")

    idx = np.arange(0, focal_series.shape[0], step)
    n = len(idx)
    ncols = ncols or math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    if shared_scale:
        plot_kwargs.setdefault("vmin", focal_series[idx].min())
        plot_kwargs.setdefault("vmax", focal_series[idx].max())

    # no coordinates given -> pixel ticks on every panel are just clutter
    has_coords = any(plot_kwargs.get(k) is not None for k in ("x", "y", "extent"))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * figsize_per_panel,
                                      nrows * figsize_per_panel),
                             squeeze=False)
    axes = axes.ravel()

    for ax, i in zip(axes, idx):
        title = f"Δf = {defoci[i]:.0f}nm" if defoci is not None else f"#{i}"
        plot_2d(focal_series[i], ax=ax, title=title, colorbar=False, **plot_kwargs)
        if not has_coords:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax in axes[n:]:          # hide unused panels
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, axes[:n]