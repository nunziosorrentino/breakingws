# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Nunziato Sorrentino (nunziato.sorrentino@pi.infn.it)
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
import six
import numpy as np

from breakingws import BREAKINGWS_DATA_RR
from breakingws.datamanage.imagesmanager import ImagesManager

class TestImages(unittest.TestCase):

    """Unit test for imagesmanager module"""
    
    def test_labels(self):
        """Unit test that verifies the correct split of labels 
           between images.
        """
        print('AAAAAA', BREAKINGWS_DATA_RR)
        test_images = ImagesManager.from_directory('../breakingws/datamanage/data')
        self.assertEqual(list(test_images.dict_imgs.keys()), 
                         test_images.images_ids)
        m1=[any([1, 0, 0, 0])==any(i) for i in test_images.labels[:100]]
        m2=[any([0, 1, 0, 0])==any(i) for i in test_images.labels[100:200]]
        m3=[any([0, 0, 1, 0])==any(i) for i in test_images.labels[200:300]]
        m4=[any([0, 0, 0, 1])==any(i) for i in test_images.labels[300:]]
        print('QUIIII', test_images.images_ids)
        self.assertTrue(any(m1)) 
        self.assertTrue(any(m2)) 
        self.assertTrue(any(m3)) 
        self.assertTrue(any(m4))                          

if __name__ == '__main__':
    unittest.main()
