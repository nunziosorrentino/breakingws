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
import glob

from breakingws import BREAKINGWS_DATA_RR

class TestGlob(unittest.TestCase):

    """Unit test for using glob module"""
    
    def test_look_at(self):
        """
        """
        print(glob.glob(BREAKINGWS_DATA_RR))

if __name__ == '__main__':
    unittest.main()

        
