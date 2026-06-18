from __future__ import annotations

from pathlib import Path

import numpy as np


def load_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column text spectrum: frequency and intensity."""
    data = np.loadtxt(Path(path), comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected at least two columns: frequency and intensity.")
    frequency = np.asarray(data[:, 0], dtype=float)
    intensity = np.asarray(data[:, 1], dtype=float)
    _validate_frequency(frequency)
    return frequency, intensity


def load_sed(
    frequency_path: str | Path,
    sed_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a frequency vector and an SED matrix arranged as frequency x q-point."""
    frequency = np.asarray(np.loadtxt(Path(frequency_path)), dtype=float).reshape(-1)
    sed = np.asarray(np.loadtxt(Path(sed_path)), dtype=float)
    if sed.ndim == 1:
        sed = sed[:, None]
    if sed.shape[0] != frequency.size and sed.shape[1] == frequency.size:
        sed = sed.T
    if sed.shape[0] != frequency.size:
        raise ValueError(
            f"SED shape {sed.shape} is incompatible with {frequency.size} frequencies."
        )
    _validate_frequency(frequency)
    return frequency, sed


def _validate_frequency(frequency: np.ndarray) -> None:
    if frequency.size < 3:
        raise ValueError("At least three frequency points are required.")
    if not np.all(np.isfinite(frequency)):
        raise ValueError("Frequency values must be finite.")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("Frequency values must be strictly increasing.")
