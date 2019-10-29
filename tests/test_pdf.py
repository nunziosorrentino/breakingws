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

        
