from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fitting import fit_lorentz_local, lorentzian
from .io import load_spectrum
from .plotting import plot_fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phononfit")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Fit a reproducible synthetic spectrum")
    demo.add_argument("--output", type=Path, default=Path("examples/demo_fit.png"))

    fit = commands.add_parser("fit", help="Fit a two-column text spectrum")
    fit.add_argument("spectrum", type=Path)
    fit.add_argument("--center", type=float, help="Initial peak center; defaults to the maximum")
    fit.add_argument("--plot", type=Path)
    return parser


def synthetic_spectrum(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frequency = np.linspace(0.5, 3.5, 601)
    clean = lorentzian(frequency, center=1.82, hwhm=0.075, amplitude=1.6, baseline=0.08)
    intensity = clean + rng.normal(0.0, 0.025, frequency.size)
    return frequency, intensity


def result_dict(fit) -> dict[str, float]:
    return {
        "center_THz": fit.center,
        "hwhm_THz": fit.hwhm,
        "r2": fit.r2,
        "rmse_normalized": fit.rmse,
        "snr": fit.snr,
        "scattering_rate_ps^-1": fit.scattering_rate_ps_inv,
        "lifetime_ps": fit.lifetime_ps,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        frequency, intensity = synthetic_spectrum()
        fit = fit_lorentz_local(frequency, intensity, center_guess=1.82)
        plot_fit(frequency, intensity, fit, args.output, title="Synthetic SED peak")
        print(json.dumps(result_dict(fit), indent=2))
        print(f"wrote {args.output}")
        return

    frequency, intensity = load_spectrum(args.spectrum)
    center = args.center if args.center is not None else float(frequency[np.argmax(intensity)])
    fit = fit_lorentz_local(frequency, intensity, center_guess=center)
    print(json.dumps(result_dict(fit), indent=2))
    if args.plot:
        plot_fit(frequency, intensity, fit, args.plot)
        print(f"wrote {args.plot}")
