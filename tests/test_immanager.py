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
import numpy as np

from breakingws.datamanage.imagesmanager import ImagesManager

array1 = np.ones((10, 64, 64, 3))
array0 = np.zeros((10, 64, 64, 3))

d_ims = dict(zeros=array0, ones=array1)

test_images = ImagesManager(d_ims)

class TestImages(unittest.TestCase):

    """Unit test for imagesmanager module"""
    
    def test_labels(self):
        """Unit test that verifies the correct split of labels 
           between images.
        """
        self.assertEqual(list(test_images.dict_imgs.keys()), 
                         test_images.images_ids)
        m1=[any([1, 0])==any(i) for i in test_images.labels[:10]]
        m2=[any([0, 1])==any(i) for i in test_images.labels[10:]]
        self.assertTrue(any(m1)) 
        self.assertTrue(any(m2)) 
        
    def test_random(self):   
        """Unit test that verifies if the images and the labels are shaked
           in the same way.
        """   
        test_images.set_random()
        if test_images.images_ids[0] == 'zeros':
            for i, l in test_images:
                if l[0]==1:
                    self.assertEqual(i[0][0][0], 0.)
                if l[1]==1:
                    self.assertEqual(i[0][0][0], 1.) 
        if test_images.images_ids[0] == 'ones':
            for i, l in test_images:
                if l[0]==1:
                    self.assertEqual(i[0][0][0], 1.)
                if l[1]==1:
                    self.assertEqual(i[0][0][0], 0.)

if __name__ == '__main__':
    unittest.main()
