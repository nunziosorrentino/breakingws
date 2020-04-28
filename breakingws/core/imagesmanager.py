# Copyright (C) 2020 Nunzio Sorrentino (nunziato.sorrentinoi@pi.infn.it)
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

import numpy as np

"""
This is a module containing all you need for a correct images management
(helpfull storage, user friendly methods for reading/writing and be able 
to make easy proper labels)

"""

class ImagesManager:

    """
    This is a base class containing some methods and utilities
    for the manipulation of many kind of images.
    """

    def __init__(images, images_id, labels):
        """Image base constructor.
        """
        self.images = images
        self.images_id = images_id
        self.labels = labels
    
     def __len__(self):
        """Return the lenght of the array representing the images.
           This is basicaly the total number of images.
        """
        return len(self.images)  

    def __iter__(self):
        """This magic method makes class instances iterable over
           the images set.
        """
        return self.images 

    def __getitem__(self, images_id):
        return self.images[images_id]
        




