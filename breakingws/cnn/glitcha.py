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
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,\
                                    MaxPooling2D,Flatten
from tensorflow.keras.models import Model

def glitcha_model(shape, classes=4, dprate=0.25, minfilts=16):
    """Model with a high number of layers (16), aimed to classify glitches.    
    """

	inputs = Input(shape=shape) 

	# Block 1
	hidden = Conv2D(minfilts,(3,3), activation='relu')(inputs) 
	hidden = Conv2D(minfilts*2,(3,3), activation='relu')(hidden) 
	hidden = MaxPooling2D((2,2))(hidden)
	hidden = Dropout(dprate)(hidden) 
	# Block 2
	hidden = Conv2D(minfilts*4,(3,3), activation='relu')(hidden) 
	hidden = MaxPooling2D((2,2))(hidden)  
	hidden = Conv2D(minfilts*4,(3,3), activation='relu')(hidden) 
	hidden = MaxPooling2D((2,2))(hidden)
	hidden = Dropout(dprate)(hidden) 

	# Block 3 
	hidden = Conv2D(minfilts*8,(3,3), activation='relu')(hidden)
	hidden = MaxPooling2D((2,2))(hidden)
	hidden = Conv2D(minfilts*8,(3,3), activation='relu')(hidden) 
	hidden = MaxPooling2D((2,2))(hidden)
	hidden = Dropout(dprate)(hidden)

	# Block 4
	hidden = Flatten()(hidden)
	hidden = Dense(minfilts*8*4, activation='relu')(hidden) 
	hidden = Dropout(dprate)(hidden) 
    
    # Block 4
	outputs = Dense(classes, activation='softmax')(hidden) 
	# 'softmax' is more efficient for multiclass
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
	return model
