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
# -----------------------------------------------------#
#                    Library imports                   #
# -----------------------------------------------------#
import pandas as pd
import tensorflow as tf
from PIL import Image
# AUCMEDI libraries
from aucmedi import DataGenerator, NeuralNetwork
from aucmedi.data_processing.subfunctions import Padding
from stain_normalization import StainNormalization

# -----------------------------------------------------#
#                   AUCMEDI Pipeline                   #
# -----------------------------------------------------#
COL_NAMES = ["A_S", "A_D", "R", "G3", "G4", "G5"]

def run_aucmedi(x, architecture, config):
    # Define Subfunctions
    target_image = Image.open("stainnormalize_target.png")
    sf_list = [Padding(mode="square"), StainNormalization(target_image)]

    # Initialize model
    model = NeuralNetwork(
        config["nclasses"],
        channels=3,
        architecture="2D.ConvNeXtLarge"
    )

    # Load model
    model.model.load_weights(architecture)

    # Initialize Data Generator
    gen = DataGenerator(
        x,
        config["path_images"],
        img_aug=None,
        shuffle=False,
        subfunctions=sf_list,
        resize=model.meta_input,
        standardize_mode=model.meta_standardize,
        grayscale=False,
        prepare_images=False,
        sample_weights=None,
        seed=123,
        image_format=config["image_format"],
    )

    # generate predictions
    preds = model.predict(gen)
    # create dataframe from predictions. Order is relevant here ans is the same as training.
    df = pd.DataFrame(preds, columns=COL_NAMES)
    df["sample"] = x
    df["class"] = df[COL_NAMES].idxmax(axis=1)
    # Garbage collection
    del gen
    del model
    del preds
    tf.keras.backend.clear_session()
    return df
