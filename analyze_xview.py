from PIL import Image
# import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
import tfr_util as tfr
import aug_util as aug
import csv
import re

# set a logger
logger = logging.getLogger("analyzexView")


"""
  A script that analyzes a set of xView crops with corresponding metadata.txt file. 
  Args:
	pos_image_folder: A folder path to the directory storing nwpu .jpg files (positive images)
		ie ("nwpu/NWPU_VHR-10_dataset/positive_image_set/")

	metadata_file: A file path to the metadata.txt file that gives the ground truth labels for the images in pos_image_folder

	output_file: A file path to write the information to

  Outputs:
	Prints out info
"""


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--metadata_file", help="Filepath to folder containing ground truth .txt files ", required=True)
	parser.add_argument("--debug", help="Debug mode", default=False, action='store_true')
	args = parser.parse_args()

	log_level = logging.INFO
	if args.debug:
		log_level = logging.DEBUG
	logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	#logger.info("Command line: %s" % " ".join(sys.argv))
	logger.info(args)
	# logger.info("Tensorflow version: %s" % tf.pywrap_tensorflow.__version__)

	# if args.pos_image_folder:
	# 	if not os.path.exists(args.pos_image_folder):
	# 		raise ValueError("--pos_image_folder=%s does not exist.", args.pos_image_folder)

	# 	if not os.path.isdir(args.pos_image_folder):
	# 		raise ValueError("--pos_image_folder=%s is not a directory.", args.pos_image_folder)

	if args.metadata_file:
		if not os.path.exists(args.metadata_file):
			raise ValueError("--metadata_file=%s does not exist.", args.metadata_file)


	line_counter = 0
	airplane_x_coord = []
	airplane_y_coord = []
	airplane_area_norm = []
	airplanes_per_image = []
	# loop through the lines in the metadata file
	with open(args.metadata_file, 'r') as metadata_file:
		line = metadata_file.readline()
		while line:
			line_counter += 1
			if line_counter % 100 == 0:
				logger.info("Read in {} lines".format(line_counter))

			fields = line.strip().split('\t')

			image_id = fields[0]
			timestamp = fields[1]
			image_path = fields[2]
			num_objects = int(fields[3])
			
			airplanes_per_image.append(num_objects)
			
			
			try:
				int(image_id)
			except ValueError:
				raise ValueError("Invalid image_id=%s at line %s" % (image_id, i))
			
			try:
				float(timestamp)
			except ValueError:
				raise ValueError("Invalid timestamp=%s at line %s" % (timestamp, i))

			
			assert len(fields)==(5*num_objects+4), "Insufficient data items in %s at line %d" % (metadata_file_path, i)
			


			# normalize object labels     
			for j in range(num_objects): 
				obj_idx = 4 + 5*j
				class_label = fields[obj_idx]

				#["Fixed-wing_Aircraft", "Cargo_Plane", "Small_Aircraft"]
								   
				for k in range(4):
					try:
						float(fields[obj_idx+1+k])
					except ValueError:
						raise ValueError("Invalid object data=%s at line %s" % (fields[obj_idx+1+k], i))
								   
				#labels.append(class_id)
				#labels.append(fields[obj_idx+1])
				#labels.append(fields[obj_idx+2])
				#labels.append(fields[obj_idx+3])
				#labels.append(fields[obj_idx+4])
				airplane_w_norm = float(fields[obj_idx+3])
				airplane_h_norm = float(fields[obj_idx+4])
				airplane_area_norm.append(airplane_w_norm * airplane_h_norm)


			line = metadata_file.readline()


	airplanes_per_image = np.array(airplanes_per_image)
	total_num_airplanes = np.sum(airplanes_per_image)
	logger.info("Read in %d lines and found %d airplanes" % (line_counter, total_num_airplanes))

	logger.info("Number of images w/ airplanes:  %d" % (line_counter))
	logger.info("Number of airplanes:            %d" % (total_num_airplanes))
	logger.info("       mean airplanes/pos img:  %f" % (np.mean(airplanes_per_image)))
	logger.info("        std airplanes/pos img:  %f" % (np.std(airplanes_per_image)))
	logger.info("        max airplanes/pos img:  %d" % (np.max(airplanes_per_image)))

	airplane_area_norm = np.array(airplane_area_norm)
	
	logger.info("     mean area (normalized):    %f" % (np.mean(airplane_area_norm)))
	logger.info("      std area (normalized):     %f" % (np.std(airplane_area_norm)))
	#logger.info("     mean area (pixels):        %f" % (np.mean(airplane_area_pix)))
	#mean_square_pix = round(np.sqrt(np.mean(airplane_area_pix)))
	#logger.info("             or rounded:        %d x %d" % (mean_square_pix, mean_square_pix))
	#logger.info("      std area (pixels):        %f" % (np.std(airplane_area_pix)))





	logger.info("DONE")


if __name__ == "__main__":
	main()