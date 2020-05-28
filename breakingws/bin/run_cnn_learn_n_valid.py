#!/usr/bin/env python
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
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from breakingws.datamanage.glitchmanager import glts_argument_generator
from breakingws.cnn.glitcha import model

if __name__=='__main__':

    desc = \
    """Something
    """
    parser = argparse.ArgumentParser(description=desc,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-m", "--model", required=True, type=str, 
    #                    help=" ")   
    #parser.add_argument("-mv", "--modelv", type=int, default=0, 
    #                    help=" ")
    parser.add_argument("-d", "--detector", type=str, default='H1', 
                        help=" ")
    parser.add_argument("-b", "--batch", type=int, default=32, 
                        help=" ")                    
    parser.add_argument("-wc", "--wisecenter", type=bool, default=True, 
                        help=" ")
    parser.add_argument("-zr", "--zoomrange", type=float, default=0.05, 
                        help=" ")  
    parser.add_argument("-ws", "--widthshift", type=int, default=10, 
                        help=" ")
    parser.add_argument("-hs", "--heightshift", type=int, default=10, 
                        help=" ")
    parser.add_argument("-vs", "--validsplit", type=float, default=0.25, 
                        help=" ")                                                   

    options = parser.parse_args()

    # Import arguments from parser
    #model = options.model
    #modelv = options.modelv
    detector = options.detector
    batch = options.batch
    wise_center = options.wisecenter
    zoom_range = options.zoomrange
    wshift = options.widthshift
    hshift = options.heightshift
    vsplit = options.validsplit

    image_generator = ImageDataGenerator(samplewise_center=wise_center,
                                         zoom_range=zoom_range,
							             width_shift_range=wshift,
    						             height_shift_range=hshift,
    						             validation_split=vsplit,
                                         )    
    path_to_detector=os.path.join('..', 'datamanage', 
                                 'GravitySpyTrainingSetV1D1',
                                 'TrainingSetImages'+detector)
    # remenber to add ImageDataGeneretor() with validation_split                              
    data_gen = glts_argument_generator(path_to_detector, image_generator,
                                       resize=(483, 578), batch_size=batch)





