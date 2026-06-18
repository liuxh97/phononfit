# PhononFit

Small, reusable tools for spectral-energy-density (SED) analysis in atomistic
simulations.

> 🚧 **Work in progress:** v0.1 focuses on one-dimensional peak detection,
> local Lorentz fitting, fit-quality checks, and lifetime estimates.

## Current scope

- Load frequency grids and SED arrays from text files.
- Detect well-separated candidate peaks.
- Fit a local Lorentz profile without a heavy fitting dependency.
- Report center frequency, half-width at half maximum (HWHM), R², RMSE, SNR,
  scattering rate, and lifetime.
- Generate a synthetic demonstration spectrum and figure.

This public repository contains generic analysis code and synthetic data only.
It does not include unpublished calculation outputs.

![Synthetic Lorentz fit](examples/demo_fit.png?v=2)

## Installation

```powershell
git clone https://github.com/liuxh97/phononfit.git
cd phononfit
python -m pip install -e .
```

## Quick demo

```powershell
python -m phononfit demo --output examples/demo_fit.png
```

Fit a two-column text file (`frequency intensity`):

```powershell
python -m phononfit fit spectrum.dat --plot fit.png
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Layout

```text
phononfit/
|-- phononfit/
|   |-- fitting.py       # Peak detection and Lorentz fitting
|   |-- io.py            # Text-array readers
|   |-- plotting.py      # Compact diagnostic plots
|   `-- cli.py           # Demo and fit commands
|-- examples/
|-- tests/
`-- pyproject.toml
```

## Width convention

The fitter reports Lorentz HWHM in the same frequency unit as the input. For
input frequencies in THz, the optional lifetime estimate follows
`tau = 1 / (4 pi gamma)`, where `gamma` is the fitted HWHM. Users should verify
that this convention matches their SED definition before physical interpretation.

## Roadmap

- Batch fitting across q-points and temperatures.
- CSV result export and configurable quality filters.
- SED heat maps with fitted-peak overlays.
- Optional phonon-dispersion overlays.
- More line shapes and uncertainty estimates.

## License

MIT License. See [LICENSE](LICENSE).
