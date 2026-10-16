"""
Script to do linear focal series reconstruction.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import numpy as np
import matplotlib.pyplot as plt

from optics import phase_error, ctf_complex, aperture_grid, fft2d, ifft2d
from utils import *

def ctf_sum_squares(defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float) -> np.ndarray:
    """Sum the squares of the ctf functions for the different defoci."""
    sum_grid = np.zeros_like(k_grid, dtype=complex)
    for defocus in defoci:
        sum_grid += np.exp(-2j * phase_error(k_grid, wavelength, defocus, cs))
    return sum_grid


def filter_function(defocus: float, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float, reg: float=1e-12) -> np.ndarray:
    """Generate a filter function for a specific defocus in a defocus series."""
    N = len(defoci)
    aperture = aperture_grid(k_grid, wavelength, alpha_max)
    H = ctf_complex(k_grid, wavelength, defocus, cs)
    S = ctf_sum_squares(defoci, k_grid, wavelength, cs)   # no mask inside

    numerator = np.conjugate(H) - H * np.conjugate(S) / N
    denominator = N - np.abs(S) ** 2 / N + N * reg

    return numerator / denominator * aperture


def filter_function_approx(defocus: float, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float):
    """Generate an approximate filter function for a specific defocus in a defocus series."""
    N=len(defoci)
    return np.conj(ctf_complex(k_grid, wavelength, defocus, cs)) * aperture_grid(k_grid, wavelength, alpha_max) / N


def linear_reconstruction(defocus_series: np.ndarray, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float, reg: float=1e-12) -> np.ndarray:
    """Linear reconstruction of defocus series"""
    filter_functions = np.stack([
        filter_function(defocus, defoci, k_grid, wavelength, cs, alpha_max, reg)
        for defocus in defoci
    ])
    reconstructed_owf_g = np.sum(fft2d(defocus_series) * filter_functions, axis=0)
    return ifft2d(reconstructed_owf_g) + 1