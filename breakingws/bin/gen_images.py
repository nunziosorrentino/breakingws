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

import numpy as np

"""## Try to create basical image (circles, ellipses, rectangles, lines)"""

import os
import cv2

imgs_shape = (600, 800, 3)
# Generate random colors for all king of figures
def randomColor():
    return (int(np.random.rand()*128+128),
            int(np.random.rand()*128+128),int(np.random.rand()*128+128))

# Generate circles
def drawCircle(c,x,y,r):
    img = np.zeros(imgs_shape, np.uint8)
    cv2.circle(img,(x,y),r,c, -1)
    return img

def genCircle():
    im = drawCircle(randomColor(),int(np.random.uniform(300, 500)),
                    int(np.random.uniform(200, 400)), 
                    int(np.random.uniform(100, 300)))
    return im

# Generate ellispe
# temploraly with fixed angle=0
def drawEllipse(c,x,y,a,b):
    img = np.zeros(imgs_shape, np.uint8)
    cv2.ellipse(img,(x,y),(a,b), 0, 0, 360, c, -1)
    return img

def genEllipse():
    im = drawEllipse(randomColor(),int(np.random.uniform(300, 500)),
                     int(np.random.uniform(200, 400)),
                     int(np.random.uniform(200, 400)), 
                     int(np.random.uniform(100, 200)))
    return im

# Generate rectangle
def drawRectangle(c,x,y,w,h):
    img = np.zeros(imgs_shape, np.uint8)
    cv2.rectangle(img,(x,y),((x+w),(y+h)), c, -1)
    return img

def genRectangle():
    im = drawRectangle(randomColor(),int(np.random.uniform(300, 500)),
                       int(np.random.uniform(200, 400)),
                       int(np.random.uniform(100, 400)), 
                       int(np.random.uniform(100, 400)))
    return im

# Generate lines
def drawLine(c,x,y,w,h):
    img = np.zeros(imgs_shape, np.uint8)
    cv2.line(img,(x,y),((x+w),(y+h)), c, 9)
    return img

def genLine():
    im = drawLine(randomColor(), int(np.random.uniform(300, 500)),
                  int(np.random.uniform(200, 400)),
                  int(np.random.uniform(-300, 100)), 
                  int(np.random.uniform(-200, 200)))
    return im

#usefull calss for labels
def class_array(i):
    array_ = np.zeros(4)
    array_[i] = 1
    return array_

def single_lab_data(n_s):
    #now generate all data possible
    circs = np.stack([genCircle() for x in range(n_s)])
    ellips = np.stack([genEllipse() for x in range(n_s)]) 
    rects = np.stack([genRectangle() for x in range(n_s)])
    lines = np.stack([genLine() for x in range(n_s)])
    #and related labels
    c_labels = np.tile(class_array(0), (circs.shape[0], 1))
    e_labels = np.tile(class_array(1), (ellips.shape[0], 1))
    r_labels = np.tile(class_array(2), (rects.shape[0], 1))
    l_labels = np.tile(class_array(3), (lines.shape[0], 1))          
    
    return circs, ellips, rects, lines, c_labels, e_labels, r_labels, l_labels

"""## Now we create the labels for each category and randompy mix images in order to make the right training, test and validation sets"""

#usefull function for data random permutation
def permutation(n_data):
    return(np.random.permutation(n_data))
# generate data

def create_images(nsamples, random=True):
    print('Generating {} images!'.format(4*nsamples))
    circs, ellips, rects, lines, c_labels, e_labels, r_labels, l_labels = single_lab_data(nsamples)
    # Now data are splitted into three different sets
    l_ = np.concatenate((c_labels, e_labels, r_labels, l_labels))
    ###
    d_ = np.concatenate((circs, ellips, rects, lines))
    #random mix
    if random:
        permutation1 = permutation(d_.shape[0])
        l_ = l_[permutation(permutation1)]
        d_ = d_[permutation(permutation1)]
        
    return np.array(d_), np.array(l_)

if __name__ == '__main__':
    import argparse
    
    desc = \
    """Tool generating circles, rectangles, lines and ellipses for the 
       breakingws tests. The dara are saved in 
       breakingws/datamanage/data/ folder.
    """
    parser = argparse.ArgumentParser(description=desc,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--nsamples", required=True, type=int, 
                        help= "Number of images for each class.")
    parser.add_argument("-s", "--save", default=True, type=bool,
                        help= """Save or not the files in 
                        breakingws/datamanage/data/ (True or False)""")
                        
    options = parser.parse_args()
         
    save = options.save
    nsamples = options.nsamples
    
    data, labels = create_images(nsamples, random=False)
    print('Images dimentions:')
    print(data.shape)
    print('Labels dimentions:')
    print(labels.shape) 
    
    if save:
        status = True
        for i in range(nsamples):
            s = cv2.imwrite(os.path.join('..', 'datamanage', 'data', 
                                         'circles/crcs{}.png'.format(i)), 
                                          data[i])
            status = bool(s*status)
            s = cv2.imwrite(os.path.join('..', 'datamanage', 'data',
                                         'ellipses/elps{}.png'.format(i)), 
                                          data[i+nsamples])
            status = bool(s*status)
            s = cv2.imwrite(os.path.join('..', 'datamanage', 'data',
                                        'rectangles/rect{}.png'.format(i)), 
                                         data[i+2*nsamples])
            status = bool(s*status)
            s = cv2.imwrite(os.path.join('..', 'datamanage', 'data',
                                         'lines/line{}.png'.format(i)), 
                                          data[i+3*nsamples])
            status = bool(s*status)
        print('Did you save the images?')
        if status:
            print('Yes!')
        else:
            print('No!')
                     
     
                     
    





