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
from breakingws.datamanage.glitchmanager import glts_argument_generator

if __name__=='__main__':

    desc = \
    """Something
    """
    parser = argparse.ArgumentParser(description=desc,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-m", "--model", required=True, type=str, 
    #                    help=" ")   
    parser.add_argument("-mv", "--modelv", type=int, default=0, 
                        help=" ")
    parser.add_argument("-d", "--detect", type=str, default='H1', 
                        help=" ")

    options = parser.parse_args()

    # Import arguments from parser
    #model = options.model
    modelv = options.modelv
    detector = options.detect

    path_to_detector=os.path.join('..', 'datamanage', 
                                 'GravitySpyTrainingSetV1D1',
                                 'TrainingSetImages'+detector)
    data_gen = glts_argument_generator(path_to_detector, resize=(483, 578))





