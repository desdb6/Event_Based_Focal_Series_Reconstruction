"""
Script to do linear focal series reconstruction.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 14/10/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optics import phase_error, ctf_complex, ctf_complex_astig, aperture_grid, fft2d, ifft2d
from utils import *

def ctf_sum_squares(defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float) -> np.ndarray:
    """Sum the squares of the ctf functions for the different defoci."""
    sum_grid = np.zeros_like(k_grid, dtype=complex)
    for defocus in defoci:
        sum_grid += np.exp(-2j * phase_error(k_grid, wavelength, defocus, cs))
    return sum_grid


def filter_function_linear(defocus: float, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float, reg: float=1e-12) -> np.ndarray:
    """Generate a filter function for a specific defocus in a defocus series."""
    N = len(defoci)
    aperture = aperture_grid(k_grid, wavelength, alpha_max)
    H = ctf_complex(k_grid, wavelength, defocus, cs)
    S = ctf_sum_squares(defoci, k_grid, wavelength, cs)   # no mask inside

    numerator = np.conjugate(H) - H * np.conjugate(S) / N
    denominator = N - np.abs(S) ** 2 / N + N * reg

    return numerator / denominator * aperture


def filter_function_approx_linear(defocus: float, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float):
    """Generate an approximate filter function for a specific defocus in a defocus series."""
    N=len(defoci)
    return np.conj(ctf_complex(k_grid, wavelength, defocus, cs)) * aperture_grid(k_grid, wavelength, alpha_max) / N


def filter_function_fbp(defocus_x: float, defocus_y: float, k_grid: np.ndarray, kx_grid: np.ndarray, ky_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float):
    """Generate a filtered backprojection kernel for an astigmatic image. (test, not sure about the generalisation of this expression)"""
    return np.abs(kx_grid) * np.abs(ky_grid) * np.conj(ctf_complex_astig(kx_grid, ky_grid, wavelength, defocus_x, defocus_y, cs)) * aperture_grid(k_grid, wavelength, alpha_max)



def linear_reconstruction(defocus_series: np.ndarray, defoci: np.ndarray, k_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float, reg: float=1e-12) -> np.ndarray:
    """Linear reconstruction of defocus series"""
    filter_functions = np.stack([
        filter_function_linear(defocus, defoci, k_grid, wavelength, cs, alpha_max, reg)
        for defocus in defoci
    ])
    reconstructed_owf_g = np.sum(fft2d(defocus_series) * filter_functions, axis=0)
    return ifft2d(reconstructed_owf_g) + 1

def fbp_reconstruction(defocus_series: np.ndarray, defoci_x: np.ndarray, defoci_y: np.ndarray, k_grid: np.ndarray, kx_grid: np.ndarray, ky_grid: np.ndarray, wavelength: float, cs: float, alpha_max: float):
    """Filtered backprojection reconstruction of defocus series"""
    filter_functions = np.stack([
        filter_function_fbp(dfx, dfy, k_grid, kx_grid, ky_grid, wavelength, cs, alpha_max)
        for dfx, dfy in tqdm(zip(defoci_x, defoci_y), desc="Generating filter functions")
    ])
    reconstructed_owf_g = np.sum(fft2d(defocus_series) * filter_functions, axis=0)
    return ifft2d(reconstructed_owf_g) + 1