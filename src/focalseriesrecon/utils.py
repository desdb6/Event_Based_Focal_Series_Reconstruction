"""
Utils file for focal series reconstruction algorithms.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

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
    """Normalize an image to given range."""
    img = img.astype('float64')
    img -= img.min()
    img /= img.max()
    img *= (minmax[1]-minmax[0])
    img += minmax[0]
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
 
 
def plot_2d(img, mode="abs", extent=None, cmap="gray", log=False,
            vmin=None, vmax=None, colorbar=True, xlabel="", ylabel="",
            title="", ax=None, save=None):
    """Plot a 2D array, e.g. an image or a Fourier transform.
 
    img   : 2D data, real or complex
    mode  : for complex data, what to show: "abs", "phase", "real" or "imag"
    extent: (left, right, bottom, top) axis values, e.g. in nm or 1/nm
    log   : show log10(1 + value), useful for diffraction / Fourier data
    """
    img = to_numpy(img)
 
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
