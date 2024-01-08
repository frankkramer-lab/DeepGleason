DeepGleason CLI
====

DeepGleason CLI is a program that infers gleason Scores for prostata pathology slides using pretrained networks.

This program utilizes the PyVIPS library to load and store images and AUCMEDI to run the model. 
It computes a one of the following classes per 1024 x 1024 tile:
 
 - Artifact - Sponge
 - Artifact Dirt
 - Regular Tissue
 - PIN, precursor or unclear tissue
 - Gleason 3
 - Gleason 4
 - Gleason 5

It outputs a BigTIFF file of the computed classes as an overlay.
Additionally a CSV can be output containing the final selected class per tile as well as the soft labels.

Installation
-------

This repository contains `requirements.txt`. Use it to install the dependencies as such:
```sh
pip install -r requirements.txt
```
One of these dependencies is pyvips. PyVIPS may require the installation of LibVIPS on the system. Please refer to their installation guide [here](https://github.com/libvips/pyvips).

Additional Notes
-------
The CLI supports multiple inputs, but it is assumed that the names of all files are unique. If this is not the case this script will crash or overwrite files.

By default a color map is generated. If itt should be overlayed over the initial image use `--generate_overlay`.

This program stores intermediate results in a cache folder. This will usually default to the system partition.
Pathology images are usually heavily compressed and the uncompressed or recompressed intermediatries tend to take up a lot of hard drive.
If your system drive does not have a lot of space available please utilize the  `--cache` argument to proviide a different location.

Should this script crash, rerunning the same command will resume progress.

Licence
-------

GNU General Public Licence

