from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LorentzFit:
    center: float
    hwhm: float
    amplitude: float
    baseline: float
    r2: float
    rmse: float
    snr: float

    @property
    def scattering_rate_ps_inv(self) -> float:
        """Return 4*pi*HWHM for an input frequency axis in THz."""
        return float(4.0 * np.pi * self.hwhm)

    @property
    def lifetime_ps(self) -> float:
        rate = self.scattering_rate_ps_inv
        return float(1.0 / rate) if rate > 0 else float("inf")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return lorentzian(x, self.center, self.hwhm, self.amplitude, self.baseline)


def lorentzian(
    x: np.ndarray,
    center: float,
    hwhm: float,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Evaluate a Lorentz profile parameterized by its HWHM."""
    x = np.asarray(x, dtype=float)
    if hwhm <= 0:
        raise ValueError("hwhm must be positive")
    return baseline + amplitude / (1.0 + ((x - center) / hwhm) ** 2)


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    width = max(3, int(width) | 1)
    if width > values.size:
        width = max(1, values.size if values.size % 2 else values.size - 1)
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def candidate_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_separation: float = 0.1,
    threshold_sigma: float = 3.0,
    max_peaks: int = 8,
) -> list[int]:
    """Find local maxima above a robust noise threshold."""
    x, y = _validated_xy(x, y)
    smooth = moving_average(y, 7)
    baseline = float(np.percentile(smooth, 20))
    noise = 1.4826 * float(np.median(np.abs(smooth - np.median(smooth))))
    noise = max(noise, float(np.max(np.abs(smooth))) * 1e-8, 1e-15)
    peaks = [
        index
        for index in range(1, smooth.size - 1)
        if smooth[index] > smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and smooth[index] > baseline + threshold_sigma * noise
    ]
    peaks.sort(key=lambda index: smooth[index], reverse=True)
    selected: list[int] = []
    for index in peaks:
        if all(abs(x[index] - x[old]) >= min_separation for old in selected):
            selected.append(index)
        if len(selected) >= max_peaks:
            break
    return sorted(selected)


def fit_lorentz_local(
    x: np.ndarray,
    y: np.ndarray,
    center_guess: float,
    *,
    half_window: float = 0.35,
    center_span: float = 0.08,
    hwhm_min: float = 0.005,
    hwhm_max: float = 0.5,
    center_steps: int = 21,
    width_steps: int = 50,
) -> LorentzFit:
    """Fit a local Lorentz profile using a stable two-parameter grid search.

    At each candidate center and width, amplitude and baseline are solved by
    linear least squares. This keeps the core routine dependency-light and
    predictable for exploratory SED analysis.
    """
    x, y = _validated_xy(x, y)
    if not hwhm_min < hwhm_max:
        raise ValueError("hwhm_min must be smaller than hwhm_max")
    mask = (x >= center_guess - half_window) & (x <= center_guess + half_window)
    xx, yy = x[mask], y[mask]
    if xx.size < 8:
        raise ValueError("The local fitting window needs at least eight points.")

    scale = max(float(np.max(np.abs(yy))), 1e-300)
    yy_normalized = yy / scale
    residual = yy_normalized - moving_average(yy_normalized, 5)
    noise = max(1.4826 * float(np.median(np.abs(residual))), 1e-8)

    best: tuple[float, float, float, float, float, float, float, float] | None = None
    centers = np.linspace(center_guess - center_span, center_guess + center_span, center_steps)
    widths = np.geomspace(hwhm_min, hwhm_max, width_steps)
    for center in centers:
        for hwhm in widths:
            shape = 1.0 / (1.0 + ((xx - center) / hwhm) ** 2)
            design = np.column_stack((shape, np.ones_like(shape)))
            amplitude_n, baseline_n = np.linalg.lstsq(design, yy_normalized, rcond=None)[0]
            if amplitude_n <= 0:
                continue
            prediction = design @ np.array([amplitude_n, baseline_n])
            residual_fit = yy_normalized - prediction
            sse = float(np.sum(residual_fit**2))
            rmse = float(np.sqrt(np.mean(residual_fit**2)))
            sst = float(np.sum((yy_normalized - np.mean(yy_normalized)) ** 2))
            r2 = 1.0 - sse / sst if sst > 0 else 0.0
            snr = float(amplitude_n / noise)
            candidate = (sse, center, hwhm, amplitude_n, baseline_n, r2, rmse, snr)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("No physically valid positive-amplitude fit was found.")
    _, center, hwhm, amplitude_n, baseline_n, r2, rmse, snr = best
    return LorentzFit(
        center=float(center),
        hwhm=float(hwhm),
        amplitude=float(amplitude_n * scale),
        baseline=float(baseline_n * scale),
        r2=float(r2),
        rmse=float(rmse),
        snr=float(snr),
    )


def _validated_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise ValueError("x and y must have the same length of at least three.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    return x, y
