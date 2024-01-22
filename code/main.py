# =============================================================================#
#  Author:       Dominik Müller, Philip Meyer                                  #
#  Copyright:    2024 AG-RAIMIA-Müller, University of Augsburg                 #
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
import os
import math
import tempfile
import argparse
import logging
import pyvips
import pandas as pd
from tqdm import tqdm
from aucmedi import input_interface

from model import run_aucmedi
from proc import gen_tiles, class_reassemble

# -----------------------------------------------------#
#                     CLI Argparser                    #
# -----------------------------------------------------#
parser = argparse.ArgumentParser(description="DeepGleason: Prediction")
parser.add_argument(
    "-g",
    "--gpu",
    help="GPU ID selection for multi cluster",
    required=False,
    type=int,
    dest="gpu",
    default=0,
)

parser.add_argument(
    "--cache",
    help="The location for temporary files that care generated during generation",
    dest="cache",
    required=False,
    type=str,
)  # Make TempDir default
parser.add_argument(
    "-i",
    "--input",
    help="Path to the input slide",
    dest="input",
    action="append",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    help="Path where the slides are stored",
    dest="output",
    required=False,
    default=".",
    type=str,
)
parser.add_argument(
    "--model",
    help="Model the XAI is computed upon",
    dest="model",
    default="model.hdf5",
    required=False,
    type=str,
)

parser.add_argument(
    "--generate_overlay",
    help="merge prediction distribution with base image as overlay",
    dest="gen_overlay",
    action="store_true",
    default=False,
    required=False,
)

parser.add_argument(
    "-p",
    "--predictions",
    help="output CSV containing predicted soft labels",
    dest="prediction",
    required=False,
    type=str,
    default="predictions.csv",
)
args = parser.parse_args()


#------------------------------------------------------#
#                     Configuration                    #
#------------------------------------------------------#
logging.basicConfig(level=logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

PATCH_SIZE = (1024, 1024)

STORE_PREDICTIONS = not (args.prediction is None)
APPEND_PREDICTIONS = os.path.exists(args.prediction)
PREDICTION_PATH = args.prediction


INPUTS = args.input

RES_PATH = args.output  # location of full slides
if not os.path.exists(RES_PATH):
    os.mkdir(RES_PATH)

BASE_PATH = args.cache  # cache base path
TEMP_CACHE = False
if BASE_PATH is None:
    BASE_PATH = tempfile.TemporaryDirectory(prefix="tmp.DeepGleason.").name
    TEMP_CACHE = True

if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

MODEL = args.model


config = {}
COL_NAMES = ["A_S", "A_D", "R", "G3", "G4", "G5"]

# pyvips.cache_set_max_mem(0) #This may be necessary to cache operations. 
# On the other hand this is incredibly useful to accelerate null computations
print(
    "linked to libvips {}.{}.{}".format(
        pyvips.version(0), pyvips.version(1), pyvips.version(2)
    )
)

#------------------------------------------------------#
#                      Main Script                     #
#------------------------------------------------------#
for i, slide in enumerate(INPUTS):
    slide_name = os.path.basename(slide)
    slide_name = slide_name[: slide_name.find(".")]
    patch_path = os.path.join(BASE_PATH, slide_name)
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)
    if not os.path.exists(os.path.join(RES_PATH, slide_name + "_gleason.tiff")):
        print("Loaded Tiff:", slide)
        max_X, max_Y, xres, yres = gen_tiles(patch_path, slide, slide_name, PATCH_SIZE)
        print("Generated Patches for Model")

        # generate predictions and XAI
        io = input_interface(
            "directory",
            path_imagedir=patch_path,
            path_data=None,
            training=False,
            ohe=False,
        )
        (x, _, _, _, image_format) = io
        config["nclasses"] = len(COL_NAMES)
        config["image_format"] = image_format
        config["path_images"] = patch_path

        df_res = run_aucmedi(x, MODEL, config)

        print("aucmedi prediction completed")
        # store predictions
        if STORE_PREDICTIONS:
            if APPEND_PREDICTIONS:
                df_res = pd.concat(
                    [pd.read_csv(PREDICTION_PATH), df_res], ignore_index=True
                )
            df_res.to_csv(PREDICTION_PATH)

        df_res.set_index("sample", inplace=True)

        if not os.path.exists(os.path.join(RES_PATH, slide_name + "_gleason.tiff")):
            os.environ["VIPS_CONCURRENCY"] = "1"
            res = pyvips.Image.new_from_array(
                class_reassemble(max_X, max_Y, slide_name, df_res, PATCH_SIZE),
                interpretation="rgb",
            )
            res = res.resize(PATCH_SIZE[0], kernel="nearest", vscale=PATCH_SIZE[1])
            res = res.rot270()
            res = res.flipver()
            img = None
            if args.gen_overlay:
                img_r = pyvips.Image.new_from_file(slide, page=0)
                img_g = pyvips.Image.new_from_file(slide, page=1)
                img_b = pyvips.Image.new_from_file(slide, page=2)

                img = img_r.bandjoin([img_g, img_b])
                img = img.copy(interpretation="rgb")

                res *= 0.3
                img *= 0.7

                res += img

            props = {
                "compression": "jpeg",
                "xres": xres,
                "yres": yres,
                "tile": True,
                "tile_width": PATCH_SIZE[0],
                "tile_height": PATCH_SIZE[1],
                "pyramid": True,
                "bigtiff": True,
            }
            res.tiffsave(os.path.join(RES_PATH, slide_name + "_gleason.tiff"), **props)
            del res
    else : print("Skipping slide:", slide, "- Already output file existing!")

    # cleanup
    for f in os.listdir(patch_path):
        os.remove(os.path.join(patch_path, f))
    os.rmdir(patch_path)
if TEMP_CACHE:
    os.rmdir(BASE_PATH)
