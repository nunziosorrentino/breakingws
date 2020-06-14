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
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
#resize=(483, 578) 

def glts_augment_generator(inputpath, datagen=ImageDataGenerator(), 
                           dataframe=None, resize=None, seed=1,
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
            if resize is None:
                resize = image.shape[:-1]                                  
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


