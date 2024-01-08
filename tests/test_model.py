# ==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2023 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# ==============================================================================#
# -----------------------------------------------------#
#                    Library imports                   #
# -----------------------------------------------------#
# External libraries
import unittest
import os
from aucmedi import NeuralNetwork

# Internal libraries
from aucmedi.ensemble.aggregate import *

# -----------------------------------------------------#
#                Unittest: Model Access                #
# -----------------------------------------------------#
class DeepGleasonModels(unittest.TestCase):
    def test_model_DenseNet121_exist(self):
        path_model = os.path.join("model/DeepGleason.model.DenseNet121.hdf5")
        res = os.path.exists(path_model)
        self.assertTrue(res)

    def test_model_DenseNet121_load(self):
        path_model = os.path.join("model/DeepGleason.model.DenseNet121.hdf5")
        model = NeuralNetwork(n_labels=7, channels=3)
        model.load(path_model)
        model.model.summary()

    def test_model_ResNeXt101_exist(self):
        path_model = os.path.join("model/DeepGleason.model.ResNeXt101.hdf5")
        res = os.path.exists(path_model)
        self.assertTrue(res)

    def test_model_ResNeXt101_load(self):
        path_model = os.path.join("model/DeepGleason.model.ResNeXt101.hdf5")
        model = NeuralNetwork(n_labels=7, channels=3)
        model.load(path_model)
        model.model.summary()