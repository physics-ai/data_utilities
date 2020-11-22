"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


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

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Path to folder containing image chips (ie 'Image_Chips/' ")
    parser.add_argument("--json_filepath", help="Filepath to GEOJSON coordinate file")
    parser.add_argument("--output_folder", help="Path to folder to hold the created chips from this script")
    parser.add_argument("--output_file", help="metadata file to write from the chipped images")
    parser.add_argument("-t", "--test_percent", type=float, default=0.333,
                    help="Percent to split into test (ie .25 = test set is 25% total)")
    parser.add_argument("-s", "--suffix", type=str, default='t1',
                    help="Output TFRecord suffix. Default suffix 't1' will output 'xview_train_t1.record' and 'xview_test_t1.record'")
    parser.add_argument("-a","--augment", type=bool, default=False,
    				help="A boolean value whether or not to use augmentation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # added
    if args.input_folder:
        if not os.path.exists(args.input_folder):
            raise ValueError("--input_folder=%s does not exist.", args.input_folder)
        
        if not os.path.isdir(args.input_folder):
            raise ValueError("--input_folder=%s is not a directory.", args.input_folder)


    if args.augment==False or args.augment=='false' or args.augment=='False':
        args.augment=False
    else:
        args.augment=True

    #resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
    #sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
    #res = [(500,500),(400,400),(300,300),(200,200)]
    res = [(256,256)]
    # accepted_classes = [11,12,13]
    # accepted_classes = [40, 50, 51] # ships!
    accepted_classes = [40, 41, 42, 44, 45, 47, 49, 50, 51, 52] # Motorboat, sailboats, tugboats, fishing vessels
    #accepted_classes = [50] # Yacht
    w_h_threshold = 0.0005

    #AUGMENT = False #args.augment
    SAVE_IMAGES = False
    #train_chips = 0
    #test_chips = 0

    #Parameters
    #max_chips_per_res = 100000
    #train_writer = tf.python_io.TFRecordWriter("../../data/xview/temp_mini_train_crops/xview_train_%s.record" % args.suffix)
    #test_writer = tf.python_io.TFRecordWriter("../../data/xview/temp_mini_train_crops/xview_test_%s.record" % args.suffix)

    logger.info("Augment argument = %s" % (str(args.augment)))
    logger.info("Save images argument = %s" % str(SAVE_IMAGES))
    logger.info("Reading in labels from %s" % (args.json_filepath))
    coords,chips,classes = wv.get_labels(args.json_filepath)

    #Load the class number -> class string label map
    labels = {}
    with open('xview_class_labels.txt') as f:
        for row in csv.reader(f):
            labels[int(row[0].split(":")[0])] = row[0].split(":")[1].replace(" ", "_")

    output_counter = 0

    with open(args.output_file, "w") as output_file:
        for res_ind, it in enumerate(res):
            #tot_box = 0
            logging.info("Res: %s" % str(it))
            #ind_chips = 0

            fnames = glob.glob(args.input_folder +"/*.tif")
            fnames.sort()

            #fnames = ['/home/physicsai/sim2real/data/xview/train_images/2545.tif']

            logger.info("Reading in {} .tif images from {}".format(len(fnames), args.input_folder))
            for fname in tqdm(fnames):
                #Needs to be "X.tif", ie ("5.tif")
                #Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
                name = fname.split("/")[-1]
                original_tif_num = name.split('.')[0]

                # only get image if there are airplanes in them. This'll speed up the process
                this_image_classes = classes[chips==name]
                this_image_coords = coords[chips==name]

                coords_accepted_classes = np.zeros(len(this_image_classes), dtype=np.bool)
                for accepted_class in accepted_classes:
                    if accepted_class in classes[chips==name]:
                        coords_accepted_classes += np.array(this_image_classes)==accepted_class

                if not np.any(coords_accepted_classes):
                    continue
                # else:
                #     logger.info(fname)

                arr = wv.get_image(fname)

                if len(arr.shape) < 3:
                    continue

                this_image_accepted_classes = this_image_classes[coords_accepted_classes]
                this_image_accepted_coords = this_image_coords[coords_accepted_classes]

                
                im,box,classes_final = wv.chip_image_smart(arr,this_image_accepted_coords,this_image_accepted_classes,it)

                #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
                #im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
                #split_ind = int(im.shape[0] * args.test_percent)

                for idx, image in enumerate(im):
                    # tf_example = tfr.to_tf_example(image,box[idx],classes_final[idx])

                    # added
                    # Check if there are no objects in the crop. If so, the chip_image returns [[0,0,0,0]] and [0]. 
                    # if there is no object, don't add the image or the line to the metadata.txt

                    # Check box[idx] and classes_final[idx] here
                    box_np = np.array(box[idx])
                    classes_np = np.array(classes_final[idx])
                    # classes_np_bool = np.zeros(len(classes_np),dtype=bool)
                    # for clss in range(len(classes_np)):
                    #     if classes_np[clss] in accepted_classes:
                    #         classes_np_bool[clss] = True
                    
                    # if not np.any(classes_np_bool):
                    #     continue
                    # # take away all non-airplane rows
                    # box_np = box_np[classes_np_bool]
                    # classes_np = classes_np[classes_np_bool]
                    
                    
                    # Take away rows that have too small of width or height
                    pixel_threshold = w_h_threshold*float(it[0])
                    box_np_bool = np.ones(box_np[:,2:].shape, dtype=np.bool)
                    box_np_bool[:,0] = (box_np[:,2] - box_np[:,0]) > pixel_threshold
                    box_np_bool[:,1] = (box_np[:,3] - box_np[:,1]) > pixel_threshold
                    
                    keep_these_rows = np.all(box_np_bool, axis=1)
                    #logger.info(output_counter)
                    #logger.info(box_np_bool)
                    #logger.info(keep_these_rows)
                    #logger.info(box_np)
                    #logger.info(pixel_threshold)
                    #logger.info(" ")
                    box_np = box_np[keep_these_rows]
                    classes_np = classes_np[keep_these_rows]

                    if box_np.shape[0]==0:
                        continue

                    # now can start saving the info
                    im_path = os.path.join(args.output_folder, "tif%05d-%06d.jpg" % (int(original_tif_num), output_counter))
                    #if not np.all(box_np_bool):
                        #logger.info(im_path)
                        #logger.info(name)
                    # write to metadata file the image id, timestamp, image filename/location, number of bbs, etc.
                    output_fields = []
                    output_fields.append(str(output_counter)) # frame id
                    output_fields.append(str(output_counter/3.0)) # timestamp
                    output_fields.append(im_path) # filepath to image
                    #output_fields.append(str(number_of_bbs))
                    # count number of airplanes in image
                    num_airplanes_in_crop = box_np.shape[0]

                    output_fields.append(str(num_airplanes_in_crop))

                    # print(output_fields)
                    # print(box[idx]) #[[149. 178 208. 235.], [78. 248. 13. 256.],[etc.]]. Not normalized
                    # print(classes_final[idx]) # [89., 89., 89., 89., etc.] Can use the map to return text

                    #at_least_ones_large = False

                    for m in range(box_np.shape[0]):
                        """
                        The bounding box is in the form (x1, y1, x2, y2) where 
                            x1, y1 is the top left corner and x2, y2 are the bottom right corner of the object of interest. 
                            The values are also in absolute pixels (0, 1, ..., chip_size) and not in relative terms. 
                        Need to convert it to (x, y, w, h) where x, y are at the center of the image 
                            and w, h are the width and height
                            And make the values relative. 
                        """

                        label_str = labels[classes_np[m]]
                        output_fields.append(label_str)
                        x1_raw = box_np[m][0]
                        y1_raw = box_np[m][1]
                        x2_raw = box_np[m][2]
                        y2_raw = box_np[m][3]
                        x1 = (x1_raw + x2_raw) / 2.0
                        y1 = (y1_raw + y2_raw) / 2.0
                        w_raw = x2_raw - x1_raw
                        h_raw = y2_raw - y1_raw
                        x = x1 / float(it[0])
                        y = y1 / float(it[1])
                        w = w_raw / float(it[0])
                        h = h_raw / float(it[1])

                        assert x >= 0 and x <= 1, "x value is not within 0 and 1"
                        assert y >= 0 and y <= 1, "y value is not within 0 and 1"
                        assert w >= 0 and w <= 1, "w value is not within 0 and 1"
                        assert h >= 0 and h <= 1, "h value is not within 0 and 1"

                        #if w >= w_h_threshold and h >= w_h_threshold:
                        output_fields.append(str(x))
                        output_fields.append(str(y))
                        output_fields.append(str(w))
                        output_fields.append(str(h))

                    # save image to folder
                    PIL_image = Image.fromarray(image)
                    PIL_image.save(fp=im_path)
                    
                    # write the metadata for this frame
                    output_file.write("%s\n" % ('\t'.join(output_fields)))

                    output_counter += 1


            # if res_ind == 0:
            #     max_chips_per_res = int(ind_chips * 1.5)
            #     logging.info("Max chips per resolution: %s " % max_chips_per_res)

            # logging.info("Tot Box: %d" % tot_box)
            # logging.info("Chips: %d" % ind_chips)

        # logging.info("saved: %d train chips" % train_chips)
        # logging.info("saved: %d test chips" % test_chips)
        #train_writer.close()
        #test_writer.close()

    logger.info("saved {} chips in {} ".format(output_counter, args.output_folder)) 

if __name__ == "__main__":
    main()