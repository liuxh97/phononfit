import tempfile
import unittest
from pathlib import Path

import numpy as np

from phononfit.io import load_sed, load_spectrum


class IoTests(unittest.TestCase):
    def test_load_spectrum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "spectrum.dat")
            np.savetxt(path, np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]]))
            frequency, intensity = load_spectrum(path)
        np.testing.assert_allclose(frequency, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(intensity, [1.0, 2.0, 1.0])

    def test_load_sed_transposes_q_by_frequency_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "frequency.dat", [0.0, 1.0, 2.0])
            np.savetxt(root / "sed.dat", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            frequency, sed = load_sed(root / "frequency.dat", root / "sed.dat")
        self.assertEqual(frequency.shape, (3,))
        self.assertEqual(sed.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
