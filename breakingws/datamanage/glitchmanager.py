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
import multiprocessing as mp
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

from functools import partial
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from breakingws.datamanage.imagesmanager import ImagesManager

N_CPUS = mp.cpu_count() 

#resize=(483, 578) 

def glts_augment_generator(inputpath, datagen=ImageDataGenerator(), 
                           dataframe=None, resize=(600, 800), seed=1,
                           batch_size=32, class_mode='categorical'):
    """
    This is a function that collects glitches spectrograms 
    in three different generators (training, validation and test) 
    with the option of choosing data augmentation. 
    """
    # First choose the data set for the samplewise centering
    if datagen.samplewise_center:
        sw_set = []
        i_path = os.path.join(inputpath, 'TrainingValid')
        # Choose one image for each class
        for label in os.listdir(i_path):
            # Avoid the csv file used in 'flow from datamanager'.
            if label.endswith('.csv'):
                continue
            images_names = os.listdir(os.path.join(i_path, label))
            image = mpimg.imread(os.path.join(i_path, label, 
                                              images_names[0]))
            sw_set.append(image)
        datagen.fit(np.array(sw_set), augment=True)
    # Create the first image generator, containing the data 
    # for the final prediction   
    pred_generator = datagen.flow_from_directory(
                                  directory=os.path.join(inputpath,'Test'),
                                  target_size=resize,
                                  batch_size=1,
                                  class_mode=None,
                                  shuffle=False,
                                  seed=seed,
                                  subset='validation',
                                  )
    if dataframe is None:
        # Create validation and training generators flowing from 
        # all images in the folders
        train_generator = datagen.flow_from_directory(
                                  directory=os.path.join(inputpath,
                                                         'TrainingValid'),
                                  target_size=resize, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  class_mode=class_mode, 
                                  subset='training', 
                                  seed=seed)
        valid_generator = datagen.flow_from_directory(
                                  directory=os.path.join(inputpath, 
                                                         'TrainingValid'),
                                  target_size=resize, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  class_mode=class_mode, 
                                  subset='validation', 
                                  seed=seed)
    else:
        # Create validation and training generators flowing from 
        # the images in the folders specified by the input dataframe
        dataframe = pd.read_csv(dataframe)
        train_generator = datagen.flow_from_dataframe(dataframe, 
                                  directory=os.path.join(inputpath, 
                                                         'TrainingValid'),
                                  x_col="id", y_col="label", 
                                  target_size=resize,
                                  shuffle=True, 
                                  seed=seed, 
                                  batch_size=batch_size, 
                                  class_mode=class_mode, 
                                  subset='training')
        valid_generator = datagen.flow_from_dataframe(dataframe, 
                                  directory=os.path.join(inputpath, 
                                                         'TrainingValid'),
                                  x_col="id", y_col="label", 
                                  target_size=resize, 
                                  shuffle=True,
                                  seed=seed,
                                  batch_size=batch_size, 
                                  class_mode=class_mode, 
                                  subset='validation')
    return train_generator, valid_generator, pred_generator

class GlitchManager(ImagesManager):
    """
    This is a manager class completaly dedicated to the preprocessing of 
    spectrograms contained in *Gravity Spy dataset*. This is a project 
    glitch spectrograms were hand-labeled by citizen scientists.
 
    """
    def __init__(self, images, images_id='', duration=2.):
        """Constructor.
        """
        ImagesManager.__init__(self, images, images_id)
        self.duration = duration
    
    @staticmethod    
    def _create_glitches_dict(p_list, duration=2.):
        """
        """
        current = mp.current_process()
        print('Running:', current.name, current._identity)
        glts_dict = {}    
        for p_ in p_list:
            ipaths = os.path.join(p_, 
                                "*spectrogram_{:.1f}.png".format(duration))
            path_list = glob.glob(ipaths)
            key_label = p_.split('/')[-1]
            new_glits = [mpimg.imread(i) for i in path_list]
            try:
                glts_dict[key_label] = np.append(
                                           glts_dict[key_label], new_glits
                                           )
            except KeyError:
                glts_dict[key_label] = new_glits 
            print(current._identity, ':', 
                  len(new_glits), "spectrograms imported!!!")
        print(current.name, 'finished!')  
        return glts_dict
        
    @staticmethod
    def _pool_handler(iterable, duration=2., nproc=1):
        """
        """
        if __name__=='__main__':
            print('Starting', nproc, 'processes for data acquisition!')
            with mp.Pool(processes=nproc) as proc:
                c_glitches_dict = partial(
                                       GlitchManager._create_glitches_dict, 
                                       duration=duration)
                results = proc.map(c_glitches_dict, iterable)
        return results    
        
    @classmethod    
    def from_directory(cls, dir_path, duration=2., nproc=None):
        """
        This methods enables to collect a 'GlitchManager' instance,
        indicating the path to the directory with images divided in 
        interferometers and labels as subdirectories.
        """
        partents_path = os.path.join(dir_path, "*") 
        interfs_path_list = glob.glob(partents_path)
        full_path_list = np.array([])
        for i_ in interfs_path_list: 
            p_path = os.path.join(i_, "*") 
            p_path_list = glob.glob(p_path) 
            full_path_list = np.concatenate((full_path_list, p_path_list))
        
        full_path_list = \
                [x for x in full_path_list if not x.endswith('.csv')]     
        
        if nproc is None:
            dict_glitches = cls._create_glitches_dict(full_path_list)
            
        else:
            n_files = len(full_path_list)
            splits_list = [n_files//nproc*(i+1) for i in range(nproc-1)]
            proc_iterable = np.split(full_path_list, splits_list)
        
            res = cls._pool_handler(proc_iterable, duration, nproc)
        
            dict_glitches = {}
            for p_o in res:
                dict_glitches.update(p_o)
        
            print('DONE, using multiprocessing!')

        return cls(dict_glitches, duration=duration)
        
if __name__=='__main__':
    print('This should be test!!!')
    test_images = GlitchManager.from_directory(
                                               'GravitySpyTrainingSetV1D1',
                                                nproc=N_CPUS)
    print('LABELS:', test_images.images_ids)
    print('There are', len(test_images), 'images')
    print('But glitches are:', test_images.images.shape, 'dimentioned')
    print('and glitches are:', test_images.labels.shape, 'dimentioned')
    print(test_images.labels[0], test_images.labels[100],
          test_images.labels[200], test_images.labels[300])
    print('Glts keys are:', list(test_images.dict_imgs.keys()))       

        
