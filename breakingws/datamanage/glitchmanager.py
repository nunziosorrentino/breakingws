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

from functools import partial
from matplotlib.pyplot import imread

from breakingws.datamanage.imagesmanager import ImagesManager

N_CPUS = mp.cpu_count() - 1

def create_images_dict(p_list, duration=2.):
    print('Process running')
    imgs_dict = {}    
    for p_ in p_list:
        ipaths = os.path.join(p_, 
                             "*spectrogram_{:.1f}.png".format(duration))
        path_list = glob.glob(ipaths)
        key_label = p_.split('/')[-1]
        new_images = [imread(i) for i in path_list]
        try:
            imgs_dict[key_label] = np.append(
                                           imgs_dict[key_label], new_images
                                           )
        except KeyError:
            imgs_dict[key_label] = new_images 
        print(p_list, "images imported!!!")
    print('Done process!')  
    return imgs_dict

class GlitchManager(ImagesManager):
    """
    This is a manager class completaly dedicated to the preprocessing of 
    spectrograms contained in *Gravity Spy dataset*. This is a project 
    glitch soectrograms were hand-labeled by citizen scientists.
 
    """
    def __init__(self, images, images_id='', duration=2.):
        """
        """
        ImagesManager.__init__(self, images, images_id)
        self.duration = duration
        
    @classmethod    
    def from_directory(cls, dir_path, duration=2., nproc=1):
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

        n_files = len(full_path_list)
        splits_list = [n_files//nproc*(i+1) for i in range(nproc-1)]
        proc_iterable = np.split(full_path_list, splits_list)
        
        # Set logger
        mp.log_to_stderr()
        logger = mp.get_logger()
        logger.setLevel(logging.INFO)
        
        proc = mp.Pool(processes=nproc)
        c_images_dict=partial(create_images_dict, duration=duration)
        proc_outputs = proc.map(c_images_dict, proc_iterable)
        proc.close()
        proc.join()
        
        dict_images = {}
        for proc in proc_outputs:
            dict_images.update(proc)

        return cls(dict_images, duration=duration)
        
if __name__=='__main__':
    print('This should be test!!!')
    #test_images = GlitchManager.from_directory('GravitySpyTrainingSetV1D1',
    #                                            nproc=N_CPUS)
    #print('AAAAAAAAAAAAA', test_images.images)
    #print('BBBBBBBB', test_images.images_ids)
    #print('CCCCCCCC', test_images.labels)
    #print('DDDDDDDDDDD', test_images.dict_imgs)
    #print('EEEEEEEEE', test_images.dict_labs)        

        
