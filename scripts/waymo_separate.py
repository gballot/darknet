from array import array
import cv2
import importlib
import io
import itertools
import keras
from keras.callbacks import CSVLogger
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image as img
import shutil
import sys
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_labels
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

history = CSVLogger('kerasloss.csv', append=True, separator=';')
tf.compat.v1.enable_eager_execution()

# Parameters
folder_path = "/home/gballot/NTU/FSPT Yolo/darknet/waymo/"
train_folder ='training_0000'

# Specify your Generative Factors and their ranges chosen for 'Waymo dataset'.
# Can be an array containing an empty string [''] if you don't want to specify
# a range.

# Daylight ranges. ex: ['Day', 'Night', 'Dawn/Dusk']
daylight_ranges_train = ['Day']
daylight_ranges_test = ['Night']
# Location ranges. ex: ['location_phx', 'location_sf', 'location_other']
location_ranges_train = ['']
location_ranges_test = ['']
# Total area of vehicules ranges. ex: [[0, 1], [1, 2196], [2196, 1000000000]]
total_area_vehicule_ranges_train = ['']
total_area_vehicule_ranges_test = ['']
# Number of vehicles ranges. ex: [[0, 1], [1, 61], [61, 121], [121, 1000000]]
count_vehicules_ranges_train = ['']
count_vehicules_ranges_test = ['']
# Number of cyclists ranges. ex: [[0, 1], [1, 61], [61, 121], [121, 10000]]
count_cyclists_ranges_train = ['']
count_cyclists_ranges_test = ['']
# Number of pedestrian ranges. ex: [[0, 1], [1, 61], [61, 1000000]]
count_pedestrians_ranges_train = ['']
count_pedestrians_ranges_test = ['']
# Number of signs ranges. ex: [[0, 1], [1, 61], [61, 1000000]]
count_signs_ranges_train = ['']
count_signs_ranges_test = ['']


# Concatenation of train and test ranges
daylight_ranges = list(set(daylight_ranges_train + daylight_ranges_test))
location_ranges = list(set(location_ranges_train + location_ranges_test))
total_area_vehicule_ranges = list(set(total_area_vehicule_ranges_train \
        + total_area_vehicule_ranges_test))
count_vehicules_ranges = list(set(count_vehicules_ranges_train \
        + count_vehicules_ranges_test))
count_cyclists_ranges = list(set(count_cyclists_ranges_train \
        + count_cyclists_ranges_test))
count_pedestrians_ranges = list(set(count_pedestrians_ranges_train \
        + count_pedestrians_ranges_test))
count_signs_ranges = list(set(count_signs_ranges_train \
        + count_signs_ranges_test))


def generate_folder():
    """
    Automated code to generate folder structure for all combinations of
    generative factor for the above chosen generative factor combinations
    """

    folder_count = 0
    for s0 in daylight_ranges:
        # This done to create a folder named 'Dawn_Dusk' because name
        # with 'Dawn/Dusk' creates problem
        if(s0 == 'Dawn/Dusk'):
            s0 = 'Dawn_Dusk'
        if len(s0) > 0:
            s0 = s0 + "/"
        path = folder_path + s0
        if not os.path.exists(path):
            os.mkdir(path)
        for s1 in location_ranges:
            if len(s1) > 0:
                s1 = s1 + "/"
            path = folder_path + s0 + s1
            if not os.path.exists(path):
                os.mkdir(path)
            for s2 in total_area_vehicule_ranges:
                if len(s2) > 0:
                    s2 = str(s2[0]) + " <= total_area_vehicules < " \
                            + str(s2[1]) + "/"
                path = folder_path + s0 + s1 + s2
                if not os.path.exists(path):
                    os.mkdir(path)
                for s3 in count_vehicules_ranges:
                    if len(s3) > 0:
                        s3 = str(s3[0]) + " <= count_vehicules < " \
                                + str(s3[1]) + "/"
                    path = folder_path + s0 + s1 + s2 + s3
                    if not os.path.exists(path):
                        os.mkdir(path)
                    for s4 in count_cyclists_ranges:
                        if len(s4) > 0:
                            s4 = str(s4[0]) + " <= count_cyclists < " \
                                    + str(s4[1]) + "/"
                        path = folder_path + s0 + s1 + s2 + s3 + s4
                        if not os.path.exists(path):
                            os.mkdir(path)
                        for s5 in count_pedestrians_ranges:
                            if len(s5) > 0:
                                s5 = str(s5([0]) + " <= count_pedestrians < " \
                                        + str(s5[1]) + "/"
                                        path = folder_path + s0 + s1 + s2 + s3 + s4 + s5
                                        if not os.path.exists(path):
                                        os.mkdir(path)
                                        images_path = path + "images/"
                                        if not os.path.exists(images_path):
                                        os.mkdir(images_path)
                                        labels_path = path + "labels/"
                                        if not os.path.exists(labels_path):
                                        os.mkdir(labels_path)
                                        folder_count = folder_count + 1
    return folder_count


class Labeled_image:

    def __init__(self, count_vehicules, area_vehicules, count_pedestrian, \
            count_signs, count_cyclists, daylight, location):
        self.path = ""
        self.count_vehicules = count_vehicules
        self.area_vehicules = area_vehicules
        self.count_pedestrians = count_pedestrians
        self.count_signs = count_signs
        self.count_cyclists = count_cyclists
        self.deylight = deylight
        self.location = location




def examine_frame(frame):
    for angle in [open_dataset.CameraName.Name.FRONT, \
                  open_dataset.CameraName.Name.FRONT_LEFT, \
                  open_dataset.CameraName.Name.FRONT_RIGHT, \
                  open_dataset.CameraName.Name.SIDE_LEFT, \
                  open_dataset.CameraName.Name.SIDE_RIGHT]
        count_vehicules = 0
        count_pedestrians = 0
        count_signs = 0
        count_cyclists = 0
        for camera_label in camera_labels:
            if camera_label.name != angle:
                continue
            for label in camera_label.labels:
                if label.type == open_labels.Type.TYPE_VEHICULE:
                    count_vehicules += 1
                    total_area_vehicules += label.box.width * label.box.length
                if label.type == open_labels.Type.TYPE_PEDESTRIAN:
                    count_pedestrians += 1
                if label.type == open_labels.Type.TYPE_SIGN:
                    count_signs += 1
                if label.type == open_labels.Type.TYPE_CYCLIST:
                    count_cyclists += 1


# Code to partition images based on Generative Factors
train_folder_path = folder_path + train_folder
Sum_Average_Frame_Area_TYPE_VEHICLE = 0
Total_Average_Frame_Area_TYPE_VEHICLE = 0
Total_AREA_TYPE_VEHICLE = 0 # Sum of area occupied by vehicles in a frame
FAD_Threshold = 2000 # Threshold value for sum of area occupied by vehicles in a frame
Num_Veh_Threshold = 30 # Threshold number of vehicles in a frame
Frame_Count_TYPE_PEDESTRIAN_Threshold = 5 # number of pedestrain in an image
image_count = 0
for root, dirs, files in os.walk(train_folder_path):
    for file in files:
        if file.startswith("segment"):  # avoid license file inside the folder
            filepath = os.path.join(root, file)
            #print(filepath)
            dataset = tf.data.TFRecordDataset(filepath, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame() # instance of a frame
                frame.ParseFromString(bytearray(data.numpy()))
                if not(len(frame.camera_labels) == 0):
                 # do this for every frame -
                    Frame_Count_TYPE_PEDESTRIAN = 0
                    Frame_Count_TYPE_VEHICLE = 0
                    FAD_Vehicle = 0
                    for m in range(0, len(frame.camera_labels)): # selects  frame's front camera and label 'm'
                        if (frame.camera_labels[m].name != FRONT):
                            continue
                        if (frame.laser_labels[m].type == 2 or frame.laser_labels[m].type == 4):  # Type pedestrian and cyclist
                            Frame_Count_TYPE_PEDESTRIAN = Frame_Count_TYPE_PEDESTRIAN+1
                        if (frame.laser_labels[m].type == 1):  # Type vehicle
                            Frame_Count_TYPE_VEHICLE = Frame_Count_TYPE_VEHICLE + 1
                            FAD_Vehicle = FAD_Vehicle + frame.laser_labels[m].box.width*frame.laser_labels[m].box.width*frame.laser_labels[m].box.length
                    #print(Frame_Count_TYPE_PEDESTRIAN)
                    #print(Frame_Count_TYPE_VEHICLE)
                    #print(FAD_Vehicle)
                    for index, image in enumerate(frame.images):
                        if(index == 0): # index = 0 represents the front camera
                            imag = img.open(io.BytesIO(image.image))
                            image_count = image_count + 1
                            print(image_count)
                            #print(frame.context.stats.time_of_day)



print(head)

#######################################

def print_save_labels(indent, path):
    label_path = path + "/labels/"
    print(indent + "with open('%s' + trainfolder + str(image_count) + '.txt', 'a') as f" % label_path)
    print(indent + "    f.write(frame.laser_labels)"
    #TODO exctract boxes

num_combinations = 0

for m0 in DayLightRange:
    print("                            if(frame.context.stats.time_of_day == '%s'):" % m0)
    print("                                print(frame.context.stats.time_of_day)")
    for m1 in LocationRange:
        print("                                if(frame.context.stats.location == '%s'):"% m1)
        print("                                    print(frame.context.stats.location)")
        for m2 in FAD_Vehicle:
            x = (str(m2[0])+ str(m2[1])) if(len(m2) > 1) else str(m2[0])
            print("                                    if(%s and %s):" % (m2[0], m2[1])) if(len(m2) > 1) else print("                                    if(%s):" % m2[0])
            for m3 in Frame_Count_TYPE_VEHICLE:
                y = (str(m3[0])+ str(m3[1])) if(len(m3) > 1) else str(m3[0])
                print("                                        if( %s and %s):" % (m3[0], m3[1])) if(len(m3) > 1) else print("                                        if(%s):" % m3[0])
                for m4 in Frame_Count_TYPE_PEDESTRIAN:
                    z = (str(m4[0])+ str(m4[1])) if(len(m4) > 1) else str(m4[0])
                    print("                                            if(%s and %s):" % (m4[0], m4[1])) if(len(m4) > 1) else print("                                            if(%s):" % m4[0])
                    #print("                    filepath = folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'" %(m0,m1,x,y,z))
                    if(m0 != 'Dawn/Dusk'):
                        print("                                                imag.save(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/images/'+train_folder+str(image_count)+'.jpg')" %(m0,m1,x,y,z))
                        print("                                                print(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/images/'+train_folder+str(image_count)+'.jpg')" %(m0,m1,x,y,z))
                        indent = "                                                "
                        path = folder_path + m0 + "/" +m1 + "/" + x + "/" + y + "/" + z
                        print_save_labels(indent, path)
                    else:
                        m0_new = 'Dawn_Dusk'
                        print("                                                imag.save(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/images/'+train_folder+str(image_count)+'.jpg')" %(m0_new,m1,x,y,z))
                        print("                                                print(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/images/'+train_folder+str(image_count)+'.jpg')" %(m0_new,m1,x,y,z))
                    num_combinations = num_combinations + 1

#######################################


tail = """

tf.compat.v1.disable_eager_execution()

# code to find partition folders which are empty
import os
dirName = folder_path
#dirName = '/mnt/mydrive/Datasets/Waymo/Train_Dataset';
num_empty_partitions = 0
# Iterate over the directory tree and check if directory is empty.
for (dirpath, dirnames, filenames) in os.walk(dirName):
    if len(dirnames) == 0 and len(filenames) == 0 :
        num_empty_partitions = num_empty_partitions+1
print(num_empty_partitions)
"""

print(tail)

sys.stdout = stdout
output_file.close()
print(num_combinations)
