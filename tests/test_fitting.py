import unittest

import numpy as np

from phononfit.cli import synthetic_spectrum
from phononfit.fitting import candidate_peaks, fit_lorentz_local, lorentzian


class FittingTests(unittest.TestCase):
    def test_lorentzian_reaches_amplitude_at_center(self):
        value = lorentzian(np.array([2.0]), 2.0, 0.1, amplitude=3.0, baseline=0.5)
        self.assertAlmostEqual(float(value[0]), 3.5)

    def test_fit_recovers_synthetic_peak(self):
        frequency, intensity = synthetic_spectrum()
        fit = fit_lorentz_local(frequency, intensity, center_guess=1.82)
        self.assertAlmostEqual(fit.center, 1.82, delta=0.015)
        self.assertAlmostEqual(fit.hwhm, 0.075, delta=0.015)
        self.assertGreater(fit.r2, 0.98)
        self.assertGreater(fit.snr, 10.0)

    def test_candidate_peak_finds_dominant_peak(self):
        frequency, intensity = synthetic_spectrum()
        peaks = candidate_peaks(frequency, intensity, threshold_sigma=2.0)
        centers = frequency[peaks]
        self.assertTrue(np.any(np.abs(centers - 1.82) < 0.03))


if __name__ == "__main__":
    unittest.main()
