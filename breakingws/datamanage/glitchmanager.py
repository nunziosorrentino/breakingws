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

import numpy as np
import multiprocess

from breakingws.datamanage.imagesmanager import ImagesManager

class GlitchManager(ImagesManager):
    """
    This is a manager class completaly dedicated to the preprocessing of 
    spectrograms contained in *Gravity Spy dataset*. This is a project 
    glitch soectrograms were hand-labeled by citizen scientists.
 
    """
    #nproc=None
    def __init__(self, images, interfs, images_id='', duration=2.)
        """
        """
        ImagesManager.__init__(self, images, images_id)
        self.interfs = interfs
        self.duration = duration
        
    @classmethod    
    def from_directory(cls, dir_path, interfs, duration=2., nproc=None):
        """
        This methos enables to collect a 'GlitchManager' instance by simply
        indicating the path to the directory with images divided in 
        interferometers and labels as subdirectories.
        """
        partents_path = os.path.join(dir_path, "*") 
        p_path_list = glob.glob(partents_path)
        def create_images_dict(path_list):
            imgs_dict = {} 
            for p_ in path_list:
                ipaths = os.path.join(p_, 
                         "*spectrogram_{:.1f}.png".format(duration))
                path_list = glob.glob(ipaths)
                key_label = p_.split('/')[-1]
                imgs_dict[key_label] = [imread(i) for i in path_list]
            return imgs_dict
        proc = multiprocessing.Pool(processes=n_proc)
        proc_outputs = proc.map(create_images_dict, p_path_list)
        proc.close()
        proc.join()
        return cls(proc_outputs, interfs, duration=duration)
        
