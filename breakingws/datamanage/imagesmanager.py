# Copyright (C) 2020 Nunziato Sorrentino (nunziato.sorrentinoi@pi.infn.it)
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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import glob
import logging
import multiprocessing as mp
import numpy as np

from matplotlib.pyplot import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
This is a module containing all you need for a correct images management
(helpfull storage, user friendly methods for reading/writing and be able
to make easy proper labels)

"""

class ImagesManager:

    """
    This is a base class containing some methods and utilities
    for the manipulation of many kind of images.
    
    Parameters
    ----------
    images : np.ndarray, dict
        Set of images encapsulated a single numpy array or in
        a dictionaty with keys equal to labels.

    images_id : str
        Label containing the classification of all input images.
        If 'images' is a dictionary, images_id is not required.
               
    Attributes
    ----------
    images
    images_ids 
    labels 
    dict_imgs
    shape

    Examples
    --------
    If you have 40 images with 64x64 pixels and 3 channels,
    with 4 different labels, encapsulated in directory and subdirectories,
    the input images are converted in an array with
    shape (40, 64, 64, 3).

    Labels will be collected in (40, 4) names will be collected in the list 
    attribute 'images_id'.
    """
    def __init__(self, images, images_id=''):
        """Images base constructor.
           
        """
        assert(isinstance(images_id, str))
        if isinstance(images, np.ndarray):
            self.images = np.array(images)
            self.images_ids = [images_id]
            self.labels = np.ones((len(self.images), 1), int)
            self.dict_imgs = dict(images_id=self.images)
        if isinstance(images, dict):
            self.dict_imgs = images
            self.images_ids = list(self.dict_imgs.keys()) 
            self.images = np.array(list(self.dict_imgs.values()))
            label_matrix = np.eye(len(self.images_ids), dtype=int)
            repeats = [len(k_) for k_ in list(images.values())]
            self.labels = np.repeat(label_matrix, repeats, axis=0)
        self.shape = self.images.shape    
        self.images = self.images.reshape((self.shape[0]*self.shape[1], 
                                           self.shape[2], self.shape[3], 
                                           self.shape[4])) 
        self.shape = self.images.shape                                              

    def add_images(self, images, images_id=''):
        """Peculiar method that adds new images and related labels
           to an existing set. You can also add images with 
           a new classification .
        """
        self.images = np.append(self.images, images, axis=0)
        self.shape = self.images.shape    
         
        if images_id in self.images_ids:
            self.dict_imgs[images_id] = \
            np.concatenate(self.dict_imgs[images_id], images)
            new_labels = [self.labels[0] for i in range(len(images))]
            self.labels = np.append(self.labels, new_labels, axis=0)

        else: 
            self.images_ids += [images_id]
            self.dict_imgs[images_id] = images
            # Preprocess all labels
            self.labels = \
            np.c_[self.labels, np.zeros(len(self.labels), int)]
            # Add new labels
            new_signle_label = np.zeros(len(self.images_ids), int)
            new_single_label[-1] = 1
            new_labels = [new_single_label for i in range(len(images))]
            self.labels = np.append(self.labels, new_labels, axis=0)
            
    @staticmethod
    def _create_images_dict(p_gen):
        """
        """
        current = mp.current_process()
        print('Running:', current.name, current._identity)
        imgs_dict = {}    
        for p_ in p_gen:
            ipaths = os.path.join(p_, "*.png")
            path_gen = glob.iglob(ipaths)
            key_label = p_.split('/')[-1]
            new_images = [imread(i) for i in path_gen]
            try:
                imgs_dict[key_label] = np.append(imgs_dict[key_label], 
                                             new_images)
            except KeyError:
                imgs_dict[key_label] = new_images 
            print(current._identity, ':', len(new_images), 
                                         "images imported!")
        print(current.name, 'finished!')  
        return imgs_dict
        
    @staticmethod
    def _pool_handler(iterable, nproc=1):
        """
        """
        if __name__=='__main__':
            print('Starting', nproc, 'processes for data acquisition!')
            with mp.Pool(processes=nproc) as proc:
                results = proc.map(ImagesManager._create_images_dict, 
                                   iterable)
        return results
    
    @classmethod    
    def from_directory(cls, dir_path, nproc=None):
        """
        This method enables to collect a 'ImageManager' instance by simply
        indicating the path to the directory with images divided in labels
        as subdirectories.
        """
        
        parents_path = os.path.join(dir_path, "*") 
        p_path_gen = glob.glob(parents_path)
        
        if nproc is None:
            dict_images = cls._create_images_dict(p_path_gen)
            
        else:
            n_files = len(p_path_gen)
            splits_list = [n_files//nproc*(i+1) for i in range(nproc-1)]
            proc_iterable = np.split(p_path_gen, splits_list)
        
            res = cls._pool_handler(proc_iterable, nproc)
        
            dict_images = {}
            for p_o in res:
                dict_images.update(p_o)
        
            print('DONE, using multiprocessing!')

        return cls(dict_images)
        
    def __len__(self):
        """Return the lenght of the array representing the images.
           This is basicaly the total number of images.
        """
        return self.shape[0] 
    
    def set_random(self):
        """Randomply permutate images and labels at unison. The changes 
           will be stored in 'images' and 'labels' attributes.
        """    
        print('Starting randomization!')
        self.images = self.images[np.random.permutation(len(self))]
        self.labels = self.labels[np.random.permutation(len(self))] 
        print('Images randomly mixed!')  
        
    def get_partial(self, split, rand=True):
        """Get the training, validation and test sets, given relative
           split fraction. The images are therefore ramdomly permutated.
        """
        if rand:
            self.set_random()
        t_set, v_set, tr_set = tuple(np.split(self.images, 
                                    [int(len(self)*split), 
                                    int(len(self)*split+len(self)*split)]))
        t_lab, v_lab, tr_lab = tuple(np.split(self.labels, 
                                    [int(len(self)*split), 
                                    int(len(self)*split+len(self)*split)]))
        print('Data splitted in training, validation and test sets!')                                      
        return (tr_set, tr_lab), (v_set, v_lab), (t_set, t_lab) 
           
    def __iter__(self):
        """This magic method makes class instances iterable over
           the images and labels set.
        """
        return self.images, self.labels

    def __getitem__(self, images_key):
        """Get the set of images and labels with the same category
           (use key string) or the single data with their index 
           (use integer). 
        """
        if isinstance(images_key, str):
            return self.dict_imgs[images_key]
        if isinstance(images_key, int):
            return self.images[images_key] 
        else:
            raise KeyError("Not acceptable type for {}".format(images_key))
        
if __name__=='__main__':
    N_CPUS = mp.cpu_count()
    test_images = ImagesManager.from_directory('data',
                                               #nproc=N_CPUS
                                               )
    print('LABELS:', test_images.images_ids)
    print('There are', len(test_images), 'images')
    print('But images are:', test_images.shape, 'dimentioned')
    print('and labels are:', test_images.labels.shape, 'dimentioned')
    print(test_images.labels[0], test_images.labels[100],
          test_images.labels[200], test_images.labels[300])
    print('Imgs keys are:', list(test_images.dict_imgs.keys()))
    train_, valor_, prov_ = test_images.get_partial(0.15)
    print(len(train_[0]), len(train_[1]))
    print(len(valor_[0]), len(valor_[1]))
    print(len(prov_[0]), len(prov_[1]))
        




