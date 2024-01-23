# =============================================================================#
#  Author:       Dominik Müller, Philip Meyer                                  #
#  Copyright:    2023 AG-RAIMIA-Müller, University of Augsburg                 #
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
# =============================================================================#
#------------------------------------------------------#
#                    Library imports                   #
#------------------------------------------------------#
# External libraries
import unittest
import tempfile
import numpy as np
import os
import json
import zipfile

#------------------------------------------------------#
#                 Public Kaggle Token                 #
#------------------------------------------------------#
# public token, please do not misuse
# should be only used for the testing suite of the DeepGleason software
token = {"username": "misitlab", "key": "503c4a25182320957f0464a8551d3f48"}
# testing file
testing_file = "train_images/006f6aa35a78965c92fffd1fbd53a058.tiff"

#------------------------------------------------------#
#                   Unittest: Usage                    #
#------------------------------------------------------#
class DeepGleasonUsage(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        # Create temporary directory-based imaging data
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.DeepGleason.")
        # opens file and dumps python dict to json object
        path_kaggle_token = os.path.join(self.tmp_data.name, "kaggle.json")
        with open(path_kaggle_token, "w") as writer:
            json.dump(token, writer)
        os.environ["KAGGLE_CONFIG_DIR"] = self.tmp_data.name
        # Load kaggle api
        import kaggle
        api = kaggle.api
        # download wsi scan as testing file
        api.competition_download_file(
            competition="prostate-cancer-grade-assessment",
            file_name=testing_file,
            path=self.tmp_data.name,
        )
        # unzip downloaded wsi
        path_wsi_zip = os.path.join(self.tmp_data.name, 
                                    testing_file.split("/")[-1] + ".zip")
        with zipfile.ZipFile(path_wsi_zip) as z:
            z.extractall(self.tmp_data.name)

    #--------------------------------------------------#
    #              Run DeepGleason - Help              #
    #--------------------------------------------------#
    def test_runDeepGleason_help(self):
        path_wsi = os.path.join(self.tmp_data.name, testing_file.split("/")[-1])
        res = os.system("python code/main.py --help")
        self.assertEqual(res, 0)

    #--------------------------------------------------#
    #              Run DeepGleason - Basic             #
    #--------------------------------------------------#
    def test_runDeepGleason_basic(self):
        path_wsi = os.path.join(self.tmp_data.name, testing_file.split("/")[-1])
        path_output = os.path.join(self.tmp_data.name)
        path_preds = os.path.join(self.tmp_data.name, "preds.csv")
        path_model = os.path.join("models", "model.DenseNet121.hdf5")
        res = os.system("python code/main.py -i " + path_wsi + " -o " + \
                        path_output + " --model " + path_model + \
                        " -p " + path_preds)
        self.assertEqual(res, 0)
        self.assertTrue(os.path.exists(path_preds))
        self.assertTrue(os.path.exists(os.path.join(path_output, 
                        testing_file.split("/")[-1].split(".")[0] + \
                            "_gleason.tiff")))

    #--------------------------------------------------#
    #            Run DeepGleason - Overlay             #
    #--------------------------------------------------#
    def test_runDeepGleason_overlay(self):
        path_wsi = os.path.join(self.tmp_data.name, testing_file.split("/")[-1])
        path_output = os.path.join(self.tmp_data.name)
        path_preds = os.path.join(self.tmp_data.name, "preds.csv")
        path_model = os.path.join("models", ".model.DenseNet121.hdf5")
        res = os.system("python code/main.py -i " + path_wsi + " -o " + \
                        path_output + " --model " + path_model + \
                        " -p " + path_preds + " --generate_overlay")
        self.assertEqual(res, 0)
        self.assertTrue(os.path.exists(path_preds))
        self.assertTrue(os.path.exists(os.path.join(path_output, 
                        testing_file.split("/")[-1].split(".")[0] + \
                            "_gleason.tiff")))

    #--------------------------------------------------#
    #            Run DeepGleason - Multiple            #
    #--------------------------------------------------#
    def test_runDeepGleason_multiple(self):
        # Create temporary directory-based imaging data for multiple slides
        tmp_data_multiple= tempfile.TemporaryDirectory(prefix="tmp.DeepGleason.")
        path_wsi = os.path.join(self.tmp_data.name, testing_file.split("/")[-1])
        os.symlink(path_wsi, os.path.join(tmp_data_multiple.name, "one.tiff"))
        os.symlink(path_wsi, os.path.join(tmp_data_multiple.name, "two.tiff"))

        path_output = os.path.join(tmp_data_multiple.name)
        path_preds = os.path.join(tmp_data_multiple.name, "preds.csv")
        path_model = os.path.join("models", "model.DenseNet121.hdf5")
        res = os.system("python code/main.py -i " + tmp_data_multiple.name + " -o " + \
                        path_output + " --model " + path_model + \
                        " -p " + path_preds)
        self.assertEqual(res, 0)
        self.assertTrue(os.path.exists(path_preds))
        self.assertTrue(os.path.exists(os.path.join(path_output, "one" + "_gleason.tiff")))
        self.assertTrue(os.path.exists(os.path.join(path_output, "two" + "_gleason.tiff")))