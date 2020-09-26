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

#import packages
import tensorflow as tf
import tensorflow.keras as krs

#read versions
tf_vers = tf.__version__
k_vers = krs.__version__

class TestKerasInstallation(unittest.TestCase):

    """Unit test for asserting installation of keras package."""
    
    def test_tensorflow_version(self):
        """Unit test that verifies the tensorflow version.
        """
        self.assertEqual(tf_vers, '2.0.3',
            msg='Tensorflow {} is properly installed.'.format(tf_vers))

    def test_keras_version(self):
        """Unit test that verifies that keras is a tensorflow module.
        """
        self.assertIn('tf', k_vers,
            msg='Keras {} properly imported from tensorflow {}.'.format(k_vers, tf_vers))

if __name__ == '__main__':
    unittest.main()

