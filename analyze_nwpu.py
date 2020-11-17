from PIL import Image
import tensorflow as tf
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
logger = logging.getLogger("analyzeNWPU")


"""
  A script that analyzes the nwpu dataset. 
  Args:
	pos_image_folder: A folder path to the directory storing nwpu .jpg files (positive images)
		ie ("nwpu/NWPU_VHR-10_dataset/positive_image_set/")

	ground_truth_folder: A folder path to the location of ground truth .txt files 
		ie ("nwpu/NWPU_VHR-10_dataset/ground_truth")

	output_file: A file path to write the information to

  Outputs:
	Writes two files to the current directory containing training and test data in
		TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--pos_image_folder", help="Path to folder containing positive nwpu .jpg images", required=True)
	parser.add_argument("--ground_truth_folder", help="Filepath to folder containing ground truth .txt files ", required=True)
	parser.add_argument("--output_file", help="metadata file to write from the chipped images", required=True)
	parser.add_argument("--debug", help="Debug mode", default=False, action='store_true')
	args = parser.parse_args()

	log_level = logging.INFO
	if args.debug:
		log_level = logging.DEBUG
	logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	#logger.info("Command line: %s" % " ".join(sys.argv))
	logger.info(args)
	logger.info("Tensorflow version: %s" % tf.pywrap_tensorflow.__version__)

	if args.pos_image_folder:
		if not os.path.exists(args.pos_image_folder):
			raise ValueError("--pos_image_folder=%s does not exist.", args.pos_image_folder)

		if not os.path.isdir(args.pos_image_folder):
			raise ValueError("--pos_image_folder=%s is not a directory.", args.pos_image_folder)

	if args.ground_truth_folder:
		if not os.path.exists(args.ground_truth_folder):
			raise ValueError("--ground_truth_folder=%s does not exist.", args.ground_truth_folder)

		if not os.path.isdir(args.ground_truth_folder):
			raise ValueError("--ground_truth_folder=%s is not a directory.", args.ground_truth_folder)

	# if args.output_file:
	# 	if not os.path.exists(args.output_file):
	# 		raise ValueError("--output_file=%s does not exist.", args.output_file)

	# 	if not os.path.isdir(args.output_file):
	# 		raise ValueError("--output_file=%s is not a directory.", args.output_file)


	# Loop through all of the ground truth files. If there is an airplane in the image, find the corresponding image in the image_folder
	# record the number of images that have an airplane
	# write the bounding box info + file name + image dimensions to a new file

	fnames = glob.glob(os.path.join(args.ground_truth_folder, "*.txt"))
	fnames.sort()
	file_counter = 0
	airplane_counter = 0
	airplanes_per_img = []
	#airplane_w = [] # can use these to calculate average / variance of size of airplanes
	#airplane_h = []
	airplane_area_norm = []
	airplane_area_pix = []
	with open(args.output_file, 'w') as output_file:
		for fname in tqdm(fnames):
			file_counter += 1
			with open(fname, 'r') as txt_file:
				airplanes_in_this_im = 0
				line = txt_file.readline()
				while line:

					fields = line.strip().split(',') # e.g.,  ['(458', '164)', '(551', '243)', '4']

					# Skip lines that are only new lines
					if len(fields) == 1:
						line = txt_file.readline()
						continue
						
					# Make sure there aren't any errors with the data
					assert len(fields) == 5, "Problem with file %s" % (fname)

					output_fields = []
					# Only use the bounding boxes that are for airplanes. 1==airplane, 2==ship
					if int(fields[4]) == 2: 
						# get image as np array
						file_ending = fname.split('/')[-1]
						file_num = file_ending.replace('.txt','')
						im = Image.open(os.path.join(args.pos_image_folder, file_num+".jpg"))

						im_w, im_h = im.size

						# add image name
						output_fields.append("%03d.jpg" % (file_counter))
						x1 = fields[0].replace('(', '')
						y1 = fields[1].replace(')', '')
						x2 = fields[2].replace('(', '')
						y2 = fields[3].replace(')', '')

						obj_w = float(x2) - float(x1)
						obj_h = float(y2) - float(y1)
						obj_area_pix = obj_w * obj_h
						obj_area_norm = obj_area_pix / (im_w * im_h)

						# add image dimensions
						output_fields.append(str(im_w))
						output_fields.append(str(im_h))

						# add image (x,y) top left and bottom right
						output_fields.append(x1)
						output_fields.append(y1)
						output_fields.append(x2)
						output_fields.append(y2)

						# add object w, h
						output_fields.append(str(obj_w))
						output_fields.append(str(obj_h))

						# add object area 
						output_fields.append(str(obj_area_pix))
						output_fields.append(str(obj_area_norm))

						# write the metadata for this frame
						output_file.write("%s\n" % ('\t'.join(output_fields)))

						airplane_counter += 1
						airplanes_in_this_im += 1

						airplane_area_norm.append(obj_area_norm)
						airplane_area_pix.append(obj_area_pix)

					# current output_fields makeup:
					# 0: filename
					# 1: image w in pixels
					# 2: image h in pixels
					# 3: x1 - top left x coord in pixels
					# 4: y1 - top left y coord in pixels
					# 5: x2 - bottom right x coord in pixels
					# 6: y2 - bottom right y coord in pixels
					# 7: object width in pixels
					# 8: object height in pixels
					# 9: object area in pixels
					#10: object normalized area


					line = txt_file.readline()
				if airplanes_in_this_im > 0:
					airplanes_per_img.append(airplanes_in_this_im)

	logger.info("Read in %d files and found %d airplanes" % (file_counter, airplane_counter))

	airplanes_per_img = np.array(airplanes_per_img)
	logger.info("Number of images w/ airplanes:  %d" % (len(airplanes_per_img)))
	logger.info("Number of airplanes:            %d" % (len(airplane_area_norm)))
	logger.info("       mean airplanes/pos img:  %f" % (np.mean(airplanes_per_img)))
	logger.info("        std airplanes/pos img:  %f" % (np.std(airplanes_per_img)))
	logger.info("        max airplanes/pos img:  %d" % (np.max(airplanes_per_img)))

	airplane_area_norm = np.array(airplane_area_norm)
	
	logger.info("     mean area (normalized):    %f" % (np.mean(airplane_area_norm)))
	logger.info("      std area (normalized):     %f" % (np.std(airplane_area_norm)))
	logger.info("     mean area (pixels):        %f" % (np.mean(airplane_area_pix)))
	mean_square_pix = round(np.sqrt(np.mean(airplane_area_pix)))
	logger.info("             or rounded:        %d x %d" % (mean_square_pix, mean_square_pix))
	logger.info("      std area (pixels):        %f" % (np.std(airplane_area_pix)))

	logger.info("DONE")


if __name__ == "__main__":
	main()