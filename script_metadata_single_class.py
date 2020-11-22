# function to take in a metadata file with multi-class labels and makes them all 'ships'

import numpy as np
import csv
import os
import argparse
import io
import glob
#from tqdm import tqdm
import numpy as np
import logging


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--metadata_input", help="Metadata file to parse and edit", required=True)
	parser.add_argument("--metadata_output", help="Metadata file location to write to", required=True)
	parser.add_argument("--new_class_name", help="class label to assign to all objects in metadata file", required=True)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)


	if not os.path.exists(args.metadata_input):
		raise ValueError("metadata file %s does not exist.", args.metadata_input)

	# create dictionary
		# key: file name of the image
		# value: a list of lists of coordinates
	data_dict = {}
	line_counter = 0
	num_objects = 0
	with open(args.metadata_input, 'r') as metadata_in:
		with open(args.metadata_output, 'w') as metadata_out:
			line = metadata_in.readline()
			while line:
				line_counter += 1
				fields_out = []
				bbxs = []
				
				if line_counter % 100 == 0:
					print("Read in {} lines so far".format(line_counter))

				fields = line.strip().split('\t')

				image_id = fields[0]
				timestamp = fields[1]
				image_path = fields[2]
				num_objects = int(fields[3])

				fields_out.append(image_id)
				fields_out.append(timestamp)
				fields_out.append(image_path)
				fields_out.append(str(num_objects))
				
				try:
					int(image_id)
				except ValueError:
					raise ValueError("Invalid image_id=%s at line %s" % (image_id, i))
				
				try:
					float(timestamp)
				except ValueError:
					raise ValueError("Invalid timestamp=%s at line %s" % (timestamp, i))

				
				assert len(fields)==(5*num_objects+4), "Insufficient data items in image id %s for image %s" % (image_id, image_path)

				for j in range(num_objects):
					obj_idx = 4 + 5*j
					# class_label = fields[obj_idx]
					fields_out.append(args.new_class_name)
					fields_out.append(fields[obj_idx+1])
					fields_out.append(fields[obj_idx+2])
					fields_out.append(fields[obj_idx+3])
					fields_out.append(fields[obj_idx+4])
					num_objects += 1

				# write to file
				metadata_out.write("%s\n" % ("\t".join(fields_out)))

				line = metadata_in.readline()

	logger.info("Read in %d lines" % line_counter)
	logger.info("Read in %d objects" % num_objects)
	logger.info("Wrote new metadata file to %s with all objects as class %s" % (args.metadata_output, args.new_class_name))

	logger.info("DONE")

if __name__ == "__main__":
	main()



# python script_metadata_single_class.py --metadata_input=/Users/austinmurphy/Desktop/projects/sim2real/data/xview/ships/metadata_256crops.txt --metadata_output=/Users/austinmurphy/Desktop/projects/sim2real/data/xview/ships/metadata_256crops_allships.txt --new_class_name=ship

