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
import os
import pyvips

os.environ["VIPS_CONCURRENCY"] = "64"
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

#------------------------------------------------------#
#             Processing Utility Functions             #
#------------------------------------------------------#
def eval_cb(image, progress):
    print(f"\reval: percent = {progress.percent}", end="\n")


def gen_tiles(patch_path, slide, name, PATCH_SIZE):
    os.environ["VIPS_CONCURRENCY"] = "128"
    props = {
        "compression": "jpeg",
        "xres": 4000,
        "yres": 4000,
        "tile": True,
        "tile_width": PATCH_SIZE[0],
        "tile_height": PATCH_SIZE[1],
        "pyramid": True,
        "bigtiff": True,
    }
    img_r = pyvips.Image.new_from_file(slide, page=0)
    img_g = pyvips.Image.new_from_file(slide, page=1)
    img_b = pyvips.Image.new_from_file(slide, page=2)

    img = img_r.bandjoin([img_g, img_b])
    img = img.copy(interpretation="rgb")

    img.set_progress(True)
    # img.signal_connect('preeval', preeval_cb)
    img.signal_connect("eval", eval_cb)

    width = img.width
    height = img.height
    width = width - (width % PATCH_SIZE[0])
    height = height - (height % PATCH_SIZE[1])

    for px in range(0, width, PATCH_SIZE[0]):
        for py in range(0, height, PATCH_SIZE[1]):
            sample = name + "_%06d_%06d" % (px + PATCH_SIZE[0], py + PATCH_SIZE[1])

            location = os.path.join(patch_path, sample + ".png")

            if os.path.exists(location):
                continue
            # generate and store patch
            crp = img.crop(px, py, PATCH_SIZE[0], PATCH_SIZE[1])
            crp.write_to_file(location)
    return (width, height, img.get("xres"), img.get("yres"))


def class_reassemble(max_X, max_Y, slide_name, df_res, PATCH_SIZE):
    cntr = tqdm(
        total=math.ceil(max_X / PATCH_SIZE[0]) * math.ceil(max_Y / PATCH_SIZE[1])
    )
    small_version = np.zeros((max_X // PATCH_SIZE[0], max_Y // PATCH_SIZE[1], 3))
    for x, px in enumerate(range(PATCH_SIZE[0], max_X, PATCH_SIZE[0])):
        for y, py in enumerate(range(PATCH_SIZE[1], max_Y, PATCH_SIZE[1])):
            file_name = slide_name + "_%06d_%06d" % (px, py)

            samp = df_res.at[
                file_name, "class"
            ]  # May lead to interesting behavior, if names are non unique
            if isinstance(samp, pd.Series): samp = str(samp[0])
            if samp == "A_S":
                small_version[x, y] = [0.3, 0.3, 0.3]
            if samp == "A_D":
                small_version[x, y] = [0, 0, 0]  # White
            elif samp == "R":
                small_version[x, y] = [0, 1, 0]  # Green
            elif samp == "G3":
                small_version[x, y] = [1, 1, 0]     # Yellow
            elif samp == "G4":
                small_version[x, y] = [1, 0.5, 0]   # Orange
            elif samp == "G5":
                small_version[x, y] = [1, 0, 0]     # Red

            cntr.update(1)
    del cntr

    small_version *= 255
    small_version = small_version.astype(np.uint8)

    return small_version
