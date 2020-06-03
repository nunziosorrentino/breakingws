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
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from breakingws.datamanage.glitchmanager import glts_augment_generator
from breakingws.cnn.glitcha import cnn_model

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
    parser.add_argument("-b", "--batch", type=int, default=32, 
                        help=" ") 
    parser.add_argument("-e", "--epochs", type=int, default=10, 
                        help=" ")                                        
    parser.add_argument("-wc", "--wisecenter", type=bool, default=False, 
                        help=" ")
    parser.add_argument("-zr", "--zoomrange", type=float, default=0.05, 
                        help=" ")  
    parser.add_argument("-ws", "--widthshift", type=int, default=10, 
                        help=" ")
    parser.add_argument("-hs", "--heightshift", type=int, default=10, 
                        help=" ")
    parser.add_argument("-vs", "--validsplit", type=float, default=0.5, 
                        help=" ") 
    parser.add_argument("-sm", "--savemodel", type=bool, default=False, 
                        help=" ")
    parser.add_argument("-sp", "--savepredicts", type=bool, default=False, 
                        help=" ")                                                                     

    options = parser.parse_args()

    # Import arguments from parser
    #model = options.model
    #modelv = options.modelv
    batch = options.batch
    epochs = options.epochs
    wise_center = options.wisecenter
    zoom_range = options.zoomrange
    wshift = options.widthshift
    hshift = options.heightshift
    vsplit = options.validsplit
    savem = options.savemodel
    savepreds = options.savepreds
    
    shape = (483, 578, 3)
    resize = (483, 578)

    image_generator = ImageDataGenerator(samplewise_center=wise_center,
                                         zoom_range=zoom_range,
							             width_shift_range=wshift,
    						             height_shift_range=hshift,
    						             validation_split=vsplit,
                                         )    
    path_to_dataset = os.path.join('..', 'datamanage', 
                                    'GravitySpyTrainingSetV1D1')
    path_to_dataframe = os.path.join(path_to_dataset, 
                                'ds_O1_GravitySpy_2.0_archive_summary.csv')                         
    # remember to add ImageDataGeneretor() with validation_split                              
    t_gen, v_gen, p_gen = glts_augment_generator(path_to_dataset,
                                          image_generator,
                                          dataframe=path_to_dataframe, 
                                          resize=resize, 
                                          batch_size=batch
				                          )
    classes = len(t_gen.class_names)
    model = cnn_model(shape, classes=classes)
    model.summary()
    history = model.fit(t_gen, steps_per_epoch = t_gen.samples//batch, 
                   validation_data = v_gen, 
                   validation_steps = v_gen.samples//batch,
                   epochs=epochs)  
                   
    test_results = model.evaluate(v_gen, steps=v_gen.n//v_gen.batch_size)
    print('test loss and accuracy:', test_results)
                   
    if savem:
        model.save_weights(os.path.join('..', 'cnn', 'glitcha0.1.h5'))
        print('Saved model to {}'.format(os.path.join('breakingws', 'cnn',
                                                      'glitcha0.1.h5')))                 
        
    model.summary()  
    print(history.history.keys())
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy')
    
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (0-1)')
    
    plt.show()   
    
    if savepreds:
        p_gen.reset()
        pred=model.predict(p_gen, steps=p_gen.n//p_gen.batch_size, 
                             verbose=1)
        # here there are the predicted labels (the probabilities)
        predicted_class_indices=np.argmax(pred,axis=1)   
        labels = (t_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]              
        # save results in a csv file
        filenames=p_gen.filenames
        results=pd.DataFrame({"Filename":filenames,
                              "Predictions":predictions})
        results.to_csv(os.path.join('..', 'cnn','results_glitcha0.1.csv'),
                       cindex=False)




