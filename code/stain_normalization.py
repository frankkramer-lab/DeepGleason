#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2023 AG-RAIMIA-Müller, University of Augsburg,                #
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
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from PIL import Image
# Histolab libraries
from histolab.stain_normalizer import ReinhardStainNormalizer
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#       Subfunction class: Stain Normalization        #
#-----------------------------------------------------#
class StainNormalization(Subfunction_Base):
    """ A Subfunction class which which can be used for stain normalization
        of histopathology scans.

    Digital pathology images can present strong color differences due to diverse 
    acquisition techniques (e.g., scanners, laboratory equipment and procedures).

    Therefore, this Subfunction class applies Reinhard stain normalization based
    on a provided source image. 
    
    The provided source image should be the same for reproducible results.

    Reference:
        Reinhard, Erik, et al. “Color transfer between images.” IEEE Computer graphics and applications 21.5 (2001)
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, source_image):
        """ Initialization function for creating a StainNormalization Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            source_image (Pillow Image):    Pillow image which is used as source for the stain normalization.
        """
        # Cache source image
        self.target = source_image.convert("RGB")
        # Initialize & fit stain normalizer
        self.normalizer = ReinhardStainNormalizer()
        self.normalizer.fit(self.target)

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        image_clipped = np.clip(image, a_min=0, a_max=255)
        # Convert target image from NumPy to Pillow
        image_pillow = Image.fromarray(image_clipped).convert("RGB")
        # Apply stain normalization via histolab
        image_pillow_normalized = self.normalizer.transform(image_pillow)
        # Convert target image back from Pillow to NumPy
        image_normalized = np.asarray(image_pillow_normalized)
        # Return normalized image
        return image_normalized
