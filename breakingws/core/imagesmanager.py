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

    def __init__(self, images, labels, images_id=''):
        """Image base constructor.
        """
        assert(isinstance(images_id, str))
        self.images = images
        self.labels = labels
        self.images_id = [images_id]
        self._dict_of_imgs = dict(images_id=self.images)
        self._dict_of_labs = dict(images_id=self.labels)  

    def add_images(self, images, labels, images_id=''):
        """
        """
        self.images = np.concatenate(self.images, images)
        self.labels = np.concatenate(self.labels, labels)
        if images_id in self.images_id:
            self._dict_of_imgs[images_id] = \
            np.concatenate(self._dict_of_imgs[images_id], images)
            self._dict_of_labs[images_id] = \
            np.concatenate(self._dict_of_labs[images_id], labels)
        else: 
            self.images_id += [images_id]
            self._dict_of_imgs[images_id] = images
            self._dict_of_labs[images_id] = labels
    
    def __len__(self):
        """Return the lenght of the array representing the images.
           This is basicaly the total number of images.
        """
        return len(self.images)  

    def __iter__(self):
        """This magic method makes class instances iterable over
           the images set.
        """
        return self.images, self.labels

    def __getitem__(self, images_key):
        """
        """
        if isinstance(images_key, str):
            return self._dict_of_imgs[images_key], self._dict_of_labs[images_key]
        if isinstance(images_key, int):
            return self.images[images_key], self.labels[images_key] 
        else:
            raise KeyError("Not acceptable type for {}".format(images_key))
        




