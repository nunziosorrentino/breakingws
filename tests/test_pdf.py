import unittest
import numpy as np

from splrand.pdf import ProbabilityDensityFunction


class Testpdf(unittest.TestCase):

    """Unit test for pdf module"""
    
    def test_normalization(self, xmin=0., xmax=1.):
        """Unit test that verify pdf is normalized to one.
        """
        _x = np.linspace(xmin, xmax, 100)
        _y = 2./(xmax - xmin)**2. * (_x - xmin)
        pdf = ProbabilityDensityFunction(_x, _y)
        norm = pdf.integral(xmin, xmax)
        self.assertAlmostEqual(norm, 1.0)


if __name__ == '__main__':
    unittest.main()

        
