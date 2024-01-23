#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 AG-RAIMIA-Müller, University of Augsburg,                #
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
#              Information & System Base              #
#-----------------------------------------------------#
# Base image
FROM tensorflow/tensorflow:latest-gpu

# Meta information
LABEL authors="Dominik Müller"
LABEL contact="dominik.mueller@informatik.uni-augsburg.de"
LABEL repository="https://github.com/frankkramer-lab/DeepGleason"
LABEL license="GNU General Public License v3.0"

#-----------------------------------------------------#
#                        Setup                        #
#-----------------------------------------------------#
# Setup system environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

# Copy git repository into container
ADD . /root/DeepGleason

# Install required software dependencies (cv2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-opencv libvips-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install & Update Python pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py
RUN python3.10 -m pip install pip --upgrade

# Install DeepGleason from local git repo
RUN python3.10 -m pip install -r /root/DeepGleason/requirements.txt

# Create working directory
VOLUME ["/data"]
WORKDIR "/root/DeepGleason/"

#-----------------------------------------------------#
#                       Startup                       #
#-----------------------------------------------------#
ENTRYPOINT ["python3.10 code/main.py --input /data/ --output /data/ --model models/model.ConvNeXtBase --predictions /data/predictions.csv"]
