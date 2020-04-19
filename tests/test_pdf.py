# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Nunziato Sorrentino (nunziato.sorrentino@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import numpy as np
# this is an import just to test that keras in properly inmported (thus installed)
import tensorflow.keras as krs

from breakingws.core.pdf import ProbabilityDensityFunction

def triangular_pdf(xmin=0., xmax=1.):
    """Triangular function used for testing pdf module.
    """
    _x = np.linspace(xmin, xmax, 100)
    _y = 2./(xmax - xmin)**2. * (_x - xmin)
    _pdf = ProbabilityDensityFunction(_x, _y)
    return _pdf

class Testpdf(unittest.TestCase):

    """Unit test for pdf module"""
    
    def test_normalization(self, xmin=0., xmax=1.):
        """Unit test that verifies if the pdf is normalized to one.
        """
        norm = triangular_pdf(xmin, xmax).integral(xmin, xmax)
        self.assertAlmostEqual(norm, 1.0)

    def test_cdf(self, xmin=0., xmax=100.):
        """Unit test that checks cdf properties.
        """
        cdf = triangular_pdf(xmin, xmax).cdf
        self.assertAlmostEqual(cdf(xmin), 0.0)
        self.assertAlmostEqual(cdf(xmax), 1.0)

if __name__ == '__main__':
    unittest.main()

        
