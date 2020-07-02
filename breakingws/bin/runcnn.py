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
import ast
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from breakingws.datamanage.glitchmanager import glts_augment_generator
from breakingws.datamanage.imagesmanager import ImagesManager
from breakingws.cnn.glitcha import glitcha_model
from breakingws.cnn.imma import imma_model

cnn_models = dict(glitcha=glitcha_model,
                  imma=imma_model)

if __name__=='__main__':

    desc = \
    """This tool represents the core of BreakinGWs package. Here you can 
       use CNN models proposed, make data augmentation, make training,
       validation and test sessions and finally make a table with the 
       predictions. The model weights can be saved as well.
    """
    parser = argparse.ArgumentParser(description=desc,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="CNN model used for the learning.")
    parser.add_argument("-ag", "--augment", type=ast.literal_eval, 
                        choices=[True, False], default=False,
                        help="If True, data augmentation is applied")
    parser.add_argument("-nc", "--ncpus", type=int, default=None,
                        help="If specified, multiprocessing with 'nproc' processes are used for data acquisition. This could be necessary when data are not augmented.")                                           
    parser.add_argument("-r", "--dprate", default=0.25, type=float, 
                        help="Rate of the dropout layers")
    parser.add_argument("-mf", "--minf", default=4, type=int, 
                        help="Minimum number of filter in convolutional layers")                    
    parser.add_argument("-s", "--shape", default=None, type=tuple,
                        help= "If not None, the images are resized at 'shape' value")
    parser.add_argument("-id", "--inputdir", type=str, default='data', 
                        help="Number of directory in 'brakingws/datamanage/' folder in which the data are contained")  
    parser.add_argument("-df", "--dframe", type=str, default=None, 
                        help="If you want to select the images, name of the cvs file, contained in 'inputdir', containing the list of images that have been selected.")                                                             
    parser.add_argument("-b", "--batch", type=int, default=32, 
                        help="Batch size of the training and validation set") 
    parser.add_argument("-e", "--epochs", type=int, default=10, 
                        help="Number of epochs of the run")                                        
    parser.add_argument("-wc", "--wisecenter", type=ast.literal_eval, 
                        choices=[True, False], default=False, 
                        help="If True, augmentation divides the images by the mean over the test set")
    parser.add_argument("-zr", "--zoomrange", type=float, default=0.05, 
                        help="Zoon range on the augmentatio")  
    parser.add_argument("-ws", "--widthshift", type=int, default=10, 
                        help="Width shift of the augmentation")
    parser.add_argument("-hs", "--heightshift", type=int, default=10, 
                        help="Height shift if the augmentation")
    parser.add_argument("-vs", "--validsplit", type=float, default=0.5, 
                        help="Fraction on data used for the validation") 
    parser.add_argument("-sm", "--savemodel", type=ast.literal_eval, 
                        choices=[True, False], default=False, 
                        help="If True, model weights are saved in 'brakingws/datamanage/<outputname>'")
    parser.add_argument("-sp", "--savepreds", type=ast.literal_eval, 
                        choices=[True, False], default=False, 
                        help="If True, percentile of well predicted images are saved in 'brakingws/datamanage/<outputname>'")  
    parser.add_argument("-sf", "--savefig", type=ast.literal_eval, 
                        choices=[True, False], default=False, 
                        help="If True, the plots of training and validation performance will be saved in 'brakingws/datamanage/<outputname>'")                      
    parser.add_argument("-os", "--outputname", type=str, default='results',
                        help="Folder at which the results of this run will be saved")         
                                                                                     

    options = parser.parse_args()

    # Import arguments from parser
    model_name = options.model
    augment = options.augment
    ncpus = options.ncpus
    dprate = options.dprate
    minf = options.minf
    shape = options.shape
    inputdir = options.inputdir
    dframe = options.dframe
    batch = options.batch
    epochs = options.epochs
    wise_center = options.wisecenter
    zoom_range = options.zoomrange
    wshift = options.widthshift
    hshift = options.heightshift
    vsplit = options.validsplit
    savem = options.savemodel
    savepreds = options.savepreds
    savefig = options.savefig
    outputname = options.outputname
    
    # The set mist be contained in datamenage directory
    path_to_dataset = os.path.join('..', 'datamanage', inputdir) 

    if augment:
        resize = shape
        if shape is not None:
            shape = (shape[0], shape[1], 3)
        image_generator = ImageDataGenerator(samplewise_center=wise_center,
                                             zoom_range=zoom_range,
		    					             width_shift_range=wshift,
    	    					             height_shift_range=hshift,
    	    					             validation_split=vsplit,
                                             )                                     
        if dframe is not None:
            path_to_dataframe = os.path.join(path_to_dataset, dframe)
        else:
            path_to_dataframe = dframe        
                                     
        # Make data augmentation   
        t_gen, v_gen, p_gen = glts_augment_generator(path_to_dataset,
                                               image_generator,
                                               dataframe=path_to_dataframe, 
                                               resize=resize, 
                                               batch_size=batch
             		                           )
        if shape is None:
            single_batch = t_gen.next()
            single_image = single_batch[0].astype('uint8')
            shape = single_imahe.shape   
        classes = len(t_gen.class_indices)
                
        model = cnn_models[model_name](shape, classes, dprate, minf)
        model.summary()
        history = model.fit(t_gen, steps_per_epoch = t_gen.samples//batch, 
                       validation_data = v_gen, 
                       validation_steps = v_gen.samples//batch,
                       epochs=epochs)  
        print('Started test session:')               
        test_results = model.evaluate(v_gen, steps=v_gen.n//batch)
        print('test loss and accuracy:', test_results)
        print('Started prediction:')
        p_gen.reset()
        pred = model.predict(p_gen, steps=p_gen.n, verbose=1)
    
    if not augment:
        im_manager = ImagesManager.from_directory(path_to_dataset, ncpus) 
        t_dl, v_dl, p_dl = im_manager.get_partial(vsplit)
        classes = len(im_manager.images_ids)
        shape = im_manager.shape[1:]
        # Temporary resize not allowed in non augmentation mode 
        #resize = (shape[0], shape[1])
        model = cnn_models[model_name](shape, classes, dprate)
        model.summary() 
        history=model.fit(t_dl[0], t_dl[1], batch_size=batch,
                          validation_data=(v_dl[0], v_dl[1]), 
                          epochs=epochs)
        print('Started test session!')                               
        testresults = model.evaluate(v_dl[0], v_dl[1], batch_size=batch)
        print('test loss and accuracy:', testresults)
        print('Started prediction:')
        pred = model.predict(p_dl[0], steps=len(p_dl[0]), verbose=1)
                   
    if savem:
        model_output = os.path.join('..', 'cnn', outputname)
        model.save_weights(os.path.join(model_output, 
                           '{}weights.h5'.format(model_name)))
        print('Saved model to {}'.format(os.path.join('breakingws', 'cnn',
                                       '{}weights.h5'.format(model_name))))                 
        
    model.summary()    
    
    if savepreds:
        print('Make predictions on {} samples.'.format(p_gen.n))
        output_csvfile = os.path.join('..', 'cnn', outputname,
      'predictions_{}_{}epochs_{}minfilters_{}batches_{}dprate.csv'.format(
                        model_name, epochs, minf, batch, dprate))                            
        # here there are the predicted labels (the probabilities)
        predicted_class_indices=np.argmax(pred,axis=1)   
        labels = (t_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]              
        # save results in a csv file
        filenames=p_gen.filenames
        predicted_prob  = np.max(pred, axis=1)
        n_class = len(predicted_prob[np.array(predicted_prob) > 0.7])
        print('{} images are well classified (over 70%)'.format(n_class))
        print('Thus the {:4f}%'.format((n_class/3779)*100))
        results=pd.DataFrame({"Filename":filenames,
                              "Predictions":predictions,
                              "Probability(%)":predicted_prob*100})
        results.to_csv(output_csvfile, index=False)
        print('Results saved in {}!'.format(output_csvfile))  
        
    if savefig:
        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Categorical Crossentropy')
        plt.savefig(os.path.join('..', 'cnn', outputname,
               'loss_{}epochs_{}minfilters_{}batches_{}dprate.png').format(
                       model_name, epochs, minf, batch, dprate))
    
        plt.figure()
        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (0-1)')
        plt.savefig(os.path.join('..', 'cnn', outputname,
           'accuracy_{}epochs_{}minfilters_{}batches_{}dprate.png').format(
                       model_name, epochs, minf, batch, dprate))
                        




