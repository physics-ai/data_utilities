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
import numpy as np
import json
from tqdm import tqdm
import random

"""
xView processing helper functions for use in data processing.
"""

def scale(x,range1=(0,0),range2=(0,0)):
    """
    Linear scaling for a value x
    """
    return range2[0]*(1 - (x-range1[0]) / (range1[1]-range1[0])) + range2[1]*((x-range1[0]) / (range1[1]-range1[0]))


def get_image(fname):    
    """
    Get an image from a filepath in ndarray format
    """
    return np.array(Image.open(fname))


def get_labels(fname):
    """
    Gets label data from a geojson label file

    Args:
        fname: file path to an xView geojson label file

    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname, "r") as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes


def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0],4))
    for ind in range(coords.shape[0]):
        x1,x2 = coords[ind,:,0].min(),coords[ind,:,0].max()
        y1,y2 = coords[ind,:,1].min(),coords[ind,:,1].max()
        nc[ind] = [x1,y1,x2,y2]
    return nc


def chip_image(img,coords,classes,shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            # compute area of bb within crop 
            #area_bb_crop = (out[:,2] - out[:,0])*(out[:,3] - out[:,1])
            # compute original area of 
            #original_bb_area = (coords[:,2] - coords[:,0])*(coords[:,3] - coords[:,1])
            #original_bb_area = original_bb_area[x][y]
            # compare areas - needs to be above a threshold. Or can compute center of bounding box here
            center_obj_x = (coords[:,2] + coords[:,0])/2.0

            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            center_x = np.logical_and((center_obj_x<((i+1)*wn)),(center_obj_x>(i*wn)))
            x = x * center_x
            # (center_obj_x<((i+1)*wn)),(center_obj_x>(i*wn))

            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            center_obj_y = (out[:,3] + out[:,1])/2.0
            center_y = np.logical_and((center_obj_y<((j+1)*hn)),(center_obj_y>(j*hn)))
            y = y * center_y
            # (center_obj_y<((j+1)*hn)),(center_obj_y>(j*hn))

            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes



def check_for_objects(coords, wn, hn, i, j, w_off=0, h_off=0):
    left_edge_in_box = np.logical_and((coords[:,0]<((i+1)*wn)+w_off),(coords[:,0]>(i*wn)+w_off))
    right_edge_in_box = np.logical_and((coords[:,2]<((i+1)*wn)+w_off),(coords[:,2]>(i*wn)+w_off))

    top_edge_in_box = np.logical_and((coords[:,1]<((j+1)*hn)+h_off),(coords[:,1]>(j*hn)+h_off))
    bottom_edge_in_box = np.logical_and((coords[:,3]<((j+1)*hn)+h_off),(coords[:,3]>(j*hn)+h_off))

    return (left_edge_in_box, right_edge_in_box, top_edge_in_box, bottom_edge_in_box)

def chip_image_smart(img,coords,classes,shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format (the tif file turned into an array)
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num_raw,h_num_raw = (width/wn,height/hn)
    if float(w_num_raw).is_integer():
        w_num = int(w_num_raw)
    else:
        w_num = int(w_num_raw) + 1
    if float(h_num_raw).is_integer():
        h_num = int(h_num_raw)
    else:
        h_num = int(h_num_raw) + 1
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}

    # print("coords: \n", coords)
    seen_objects_bool = np.zeros(len(coords), dtype=np.bool) # True if seen in another crop, False if not yet seen. 

    # problems:
        # 80.tif is a good example. 
        # airplanes that are out of bounds aren't counted because the pixel for w1 is negative. 
        # plane all the way on teh right isn't used because the crop for loop doesn't include the far right and bottom edges of image. 

    # loop through and replace all negative coordinates with 0
    # also need to replace ones that are larger than the height and width...
    for i in range(len(coords)):
        for j in range(len(coords[i])):
            if j == 0:
                if coords[i][j] < 0:
                    coords[i][j] = 0
            elif j == 1:
                if coords[i][j] < 0:
                    coords[i][j] = 0
            elif j == 2:
                if coords[i][j] > width:
                    coords[i][j] = width
            elif j == 3:
                if coords[i][j] > height:
                    coords[i][j] = height
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):

            edge_bools = check_for_objects(coords=coords, wn=wn, hn=hn, i=i, j=j, w_off=0, h_off=0)
            left_edge_in_box, right_edge_in_box, top_edge_in_box, bottom_edge_in_box = edge_bools

            # this checks for partial boxes.
            x = np.logical_or( left_edge_in_box, # if left edge is in box
                               right_edge_in_box) # if right edge is in box

            y = np.logical_or( top_edge_in_box, # if top edge is in box
                               bottom_edge_in_box) # if bottom edge is in box

            both_sides_in = np.logical_and(left_edge_in_box,
                                            right_edge_in_box)
            both_top_n_bottom_in = np.logical_and(top_edge_in_box,
                                                    bottom_edge_in_box)

            object_in_box = x * y
            object_completely_in_box = both_sides_in * both_top_n_bottom_in
            object_partially_in_box = (object_in_box) & ~(object_completely_in_box)
            

            # now check if object is partially on bottom
            partial_left_edges = (~left_edge_in_box) * object_partially_in_box
            partial_right_edges = (~right_edge_in_box) * object_partially_in_box
            partial_top_edges = (~top_edge_in_box) * object_partially_in_box
            partial_bottom_edges = (~bottom_edge_in_box) * object_partially_in_box

            # can check if right and bottom have true in them. If true, add some amount... 
            # then have to check again for any partials. Might be easiest to get bool vectors from function
            # could create offset value 
            w_off, h_off = 0, 0
            if np.any(partial_right_edges):
                # calculate the width of the object that is on right edge
                partial_objs = coords[partial_right_edges]
                distance_to_edge = np.max(partial_objs[:,2] - (wn*(i+1)))
                # choose random number between distance_to_edge and int(width/3)
                if int(wn/3.0) < distance_to_edge:
                    w_off = random.randint(distance_to_edge, distance_to_edge+10)
                else:
                    w_off = random.randint(distance_to_edge, int(wn/3.0))
            if np.any(partial_bottom_edges):
                partial_objs = coords[partial_bottom_edges]
                distance_to_bottom = np.max(partial_objs[:,3] - (hn*(j+1)))
                # choose random number between distance_to_bottom and int(width/3)
                if int(hn/3.0) < np.max(distance_to_bottom):
                    h_off = random.randint(distance_to_bottom, distance_to_bottom+10)
                else:
                    h_off = random.randint(distance_to_bottom, int(hn/3.0))
            if w_off != 0 or h_off != 0:
                edge_bools = check_for_objects(coords=coords, wn=wn, hn=hn, i=i, j=j, w_off=w_off, h_off=h_off)
                left_edge_in_box, right_edge_in_box, top_edge_in_box, bottom_edge_in_box = edge_bools
                # check for any remaining partial objects. If so, don't use. 
                #
                x2 = np.logical_or( left_edge_in_box, # if left edge is in box
                                    right_edge_in_box) # if right edge is in box
                y2 = np.logical_or( top_edge_in_box, # if top edge is in box
                                    bottom_edge_in_box) # if bottom edge is in box
                both_sides_in2 = np.logical_and(left_edge_in_box,
                                                right_edge_in_box)
                both_top_n_bottom_in2 = np.logical_and(top_edge_in_box,
                                                        bottom_edge_in_box)

                object_in_box2 = x2 * y2
                object_completely_in_box2 = both_sides_in2 * both_top_n_bottom_in2
                object_partially_in_box2 = (object_in_box2) & ~(object_completely_in_box2)
                if not np.any(object_partially_in_box2):
                    # if we get in here, that means there are no more partially cropped objects. Yay! Use this crop. 
                    print("Replace!!!! Use the new image!!!!")
                    object_in_box = object_in_box2
                    # need to mark the objects as viewed if they are true here in object_in_box2

            # ~seen_objects_bool is False if the object has been seen, True if the object has not been seen. 
            # only keep objects that haven't yet been seen........
            #object_in_box = object_in_box & ~seen_objects_bool
            outn = coords[object_in_box]
            # add seen objects to list:
            seen_objects_bool = object_in_box | seen_objects_bool
            # w_off and h_off are used to give updated coordinates
            #w_off = 38
            #h_off = -54
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i) - w_off,0,wn),
                                          np.clip(outn[:,1]-(hn*j) - h_off,0,hn),
                                          np.clip(outn[:,2]-(wn*i) - w_off,0,wn),
                                          np.clip(outn[:,3]-(hn*j) - h_off,0,hn))))
            box_classes = classes[object_in_box]
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            # chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            chip_h_start = int(hn*j + h_off)
            chip_h_end = int(hn*(j+1) + h_off)
            clip_w_start = int(wn*i+w_off)
            clip_w_end = int(wn*(i+1)+w_off)

            chip = img[chip_h_start:chip_h_end, clip_w_start:clip_w_end, :3]
            # There will be chips that are smaller than the original size. Only add the appropriate size. 
            chip_shape = chip.shape
            images[k][:chip_shape[0],:chip_shape[1],:] = chip
            
            k = k + 1

    
    return images.astype(np.uint8),total_boxes,total_classes
