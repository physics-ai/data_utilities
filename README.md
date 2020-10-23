## xView Data Utilities

This repository contains data processing scripts for the xView dataset.  The script 'process_wv.py' is runnable and processes a folder containing xView imagery along with a groundtruth geojson file to create a TFRecord containing shuffled, chipped, and augmented xView patches in JPEG format.  We provide several augmentation functions in 'aug_util.py' for rotating and shifting images and bounding boxes, as well as noise injecting techniques like salt-and-pepper and gaussian blurring.  Additionally in 'wv_util.py' we provide several functions for loading, processing, and chipping xView data.

The Jupyter Notebook provided in this repository interactively illustrates an example xView processing pipeline using the provided utility functions.

## Dependencies
* Python 3, Tensorflow 1.14. See [docs/virtualenv.md](docs/virtualenv.md)

## Instructions

`process_airplane_imgs.py` can be run to chip xView images within a folder and a corresponding .geojson file. 

#### Example chipping command: 

`python process_airplane_imgs.py --input_folder /home/physicsai/sim2real/data/xview/train_images --json_filepath /home/physicsai/sim2real/data/xview/xView_train.geojson --output_folder /home/physicsai/sim2real/data/xview/temp_full_train_512crops_0.15thr output_file /home/physicsai/sim2real/data/xview/metadata_temp_full_train_512crops_0.15thr.txt`

* `--input_folder`: a path to a directory that contains xView images (.tif)
* `--json_filepath`: a path to a .geojson file containing label information for the xView images
* `--output_folder`: a path to a directory where the chipped images will be sent
* `--output_file`: a path to a new file that will be created 