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

###############################################################################
#                                Parameters                                   #
###############################################################################

folder_path = "/home/gballot/NTU/FSPT Yolo/darknet/waymo/"
train_folders = ['training_{:04d}'.format(i) for i in range(32)]
links_train_images = folder_path + "links_train_images"
links_train_labels = folder_path + "links_train_labels"
links_test_images = folder_path + "links_test_images"
links_test_labels = folder_path + "links_test_labels"

# The values are only used to be printed in the label files.
save_label_dict = {
        open_labels.Label.Type.TYPE_VEHICLE: 0,
        open_labels.Label.Type.TYPE_PEDESTRIAN: 1,
        open_labels.Label.Type.TYPE_CYCLIST: 2,
        open_labels.Label.Type.TYPE_SIGN: 3,
        open_labels.Label.Type.TYPE_UNKNOWN: 4
        }

# Specify your Generative Factors and their ranges chosen for 'Waymo dataset'.
# Can be an array containing an empty string [''] if you don't want to specify
# a range.

# Daylight ranges. ex: ['Day', 'Night', 'Dawn/Dusk']
daylight_ranges_train = ['Day']
daylight_ranges_test = ['Night']
# Location ranges. ex: ['location_phx', 'location_sf', 'location_other']
location_ranges_train = ['']
location_ranges_test = ['']
# Total area of vehicles ranges. ex: [[0, 1], [1, 2196], [2196, 1000000000]]
total_area_vehicule_ranges_train = ['']
total_area_vehicule_ranges_test = ['']
# Number of vehicles ranges. ex: [[0, 1], [1, 61], [61, 121], [121, 1000000]]
count_vehicles_ranges_train = ['']
count_vehicles_ranges_test = ['']
# Number of cyclists ranges. ex: [[0, 1], [1, 61], [61, 121], [121, 10000]]
count_cyclists_ranges_train = ['']
count_cyclists_ranges_test = ['']
# Number of pedestrian ranges. ex: [[0, 1], [1, 61], [61, 1000000]]
count_pedestrians_ranges_train = ['']
count_pedestrians_ranges_test = ['']
# Number of signs ranges. ex: [[0, 1], [1, 61], [61, 1000000]]
count_signs_ranges_train = ['']
count_signs_ranges_test = ['']


###############################################################################


# Concatenation of train and test ranges
daylight_ranges = [daylight_ranges_train, daylight_ranges_test]
location_ranges = [location_ranges_train, location_ranges_test]
total_area_vehicule_ranges = [total_area_vehicule_ranges_train, \
        total_area_vehicule_ranges_test]
count_vehicles_ranges = [count_vehicles_ranges_train, \
        count_vehicles_ranges_test]
count_cyclists_ranges = [count_cyclists_ranges_train, \
        count_cyclists_ranges_test]
count_pedestrians_ranges = [count_pedestrians_ranges_train, \
        count_pedestrians_ranges_test]
count_signs_ranges = [count_signs_ranges_train, count_signs_ranges_test]

history = CSVLogger('kerasloss.csv', append=True, separator=';')
image_id = 0
train_folder = train_folders[0]


def generate_folder():
    """
    Automated code to generate folder structure for all combinations of
    generative factor for the above chosen generative factor combinations
    """

    daylight_ranges = list(set(daylight_ranges_train + daylight_ranges_test))
    location_ranges = list(set(location_ranges_train + location_ranges_test))
    total_area_vehicule_ranges = list(set(total_area_vehicule_ranges_train \
            + total_area_vehicule_ranges_test))
    count_vehicles_ranges = list(set(count_vehicles_ranges_train \
            + count_vehicles_ranges_test))
    count_cyclists_ranges = list(set(count_cyclists_ranges_train \
            + count_cyclists_ranges_test))
    count_pedestrians_ranges = list(set(count_pedestrians_ranges_train \
            + count_pedestrians_ranges_test))
    count_signs_ranges = list(set(count_signs_ranges_train \
            + count_signs_ranges_test))
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
                    s2 = str(s2[0]) + " <= total_area_vehicles < " \
                            + str(s2[1]) + "/"
                path = folder_path + s0 + s1 + s2
                if not os.path.exists(path):
                    os.mkdir(path)
                for s3 in count_vehicles_ranges:
                    if len(s3) > 0:
                        s3 = str(s3[0]) + " <= count_vehicles < " \
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
                                s5 = str(s5[0]) + " <= count_pedestrians < " \
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


class LabeledImage:

    def __init__(self, image_id):
        self.image_id = image_id
        self.count_vehicles = 0
        self.area_vehicles = 0
        self.count_pedestrians = 0
        self.count_signs = 0
        self.count_cyclists = 0
        self.daylight = ""
        self.location = ""
        self.path_train = []
        self.path_test = []
        self.image_data = 0
        self.labels = []
        self.labels_text = ""


    def build_paths(self):
        self.path_train = []
        self.path_test = []
        for i in [0,1]:
            for s0 in daylight_ranges[i]:
                if(s0 == 'Dawn/Dusk'):
                    s0 = 'Dawn_Dusk'
                if s0 != '' and s0 != self.daylight:
                    continue
                if len(s0) > 0:
                    s0 = s0 + "/"
                path = folder_path + s0
                if not os.path.exists(path):
                    os.mkdir(path)
                for s1 in location_ranges[i]:
                    if s1 != '' and s1 != self.location:
                        continue
                    if len(s1) > 0:
                        s1 = s1 + "/"
                    path = folder_path + s0 + s1
                    if not os.path.exists(path):
                        os.mkdir(path)
                    for s2 in total_area_vehicule_ranges[i]:
                        if s2 != '':
                            if self.total_area_vehicles < s2[0] \
                                    or s2[1] <= self.total_area_vehicles:
                                continue
                        if len(s2) > 0:
                            s2 = str(s2[0]) + " <= total_area_vehicles < " \
                                    + str(s2[1]) + "/"
                        path = folder_path + s0 + s1 + s2
                        if not os.path.exists(path):
                            os.mkdir(path)
                        for s3 in count_vehicles_ranges[i]:
                            if s3 != '':
                                if self.count_vehicles < s3[0] \
                                        or s3[1] <= self.count_vehicles:
                                    continue
                            if len(s3) > 0:
                                s3 = str(s3[0]) + " <= count_vehicles < " \
                                        + str(s3[1]) + "/"
                            path = folder_path + s0 + s1 + s2 + s3
                            if not os.path.exists(path):
                                os.mkdir(path)
                            for s4 in count_cyclists_ranges[i]:
                                if s4 != '':
                                    if self.count_cyclists < s4[0] \
                                            or s4[1] <= self.count_cyclists:
                                        continue
                                if len(s4) > 0:
                                    s4 = str(s4[0]) + " <= count_cyclists < " \
                                            + str(s4[1]) + "/"
                                path = folder_path + s0 + s1 + s2 + s3 + s4
                                if not os.path.exists(path):
                                    os.mkdir(path)
                                for s5 in count_pedestrians_ranges[i]:
                                    if s5 != '':
                                        if self.count_pedestrians < s5[0] \
                                                or s5[1] <= self.count_pedestrians:
                                            continue
                                    if len(s5) > 0:
                                        s5 = str(s5[0]) + " <= count_pedestrians < " \
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
                                    image_path = images_path + train_folder \
                                            + str(self.image_id) + ".jpg"
                                    label_path = labels_path + train_folder \
                                            + str(self.image_id) + ".txt"
                                    if i == 0:
                                        self.path_train.append([image_path, \
                                                label_path])
                                    if i == 1:
                                        self.path_test.append([image_path, \
                                                label_path])

    def add_label(self, label, width, length):
        self.labels.append(label)
        label_class = save_label_dict.get(label.type)
        x = label.box.center_x / width
        y = label.box.center_y / length
        w = label.box.length / width
        h = label.box.width / length
        if len(self.labels_text) == 0:
            self.labels_text = \
                    "{0} {1:9.8f} {2:9.8f} {3:9.8f} {4:9.8f}".format(label_class, x, y, w, h)
        else:
            self.labels_text = self.labels_text \
                    + "\n{0} {1:9.8f} {2:9.8f} {3:9.8f} {4:9.8f}".format(label_class, x, y, w, h)

    def save(self):
        for path in self.path_train:
            print("save image {}".format(path[0]))
            self.image_data.save(path[0])
            with open(path[1], 'w') as f:
                f.write(self.labels_text)
        for path in self.path_test:
            print("save image {}".format(path[0]))
            self.image_data.save(path[0])
            with open(path[1], 'w') as f:
                f.write(self.labels_text)

    def add_to_links_files(self):
        with open(links_train_images, 'a') as f:
            for path in self.path_train:
                f.write(path[0] + "\n")
        with open(links_train_labels, 'a') as f:
            for path in self.path_train:
                f.write(path[1] + "\n")
        with open(links_test_images, 'a') as f:
            for path in self.path_test:
                f.write(path[0] + "\n")
        with open(links_test_labels, 'a') as f:
            for path in self.path_test:
                f.write(path[1] + "\n")




def examine_frame(frame):
    global image_id
    for angle in [open_dataset.CameraName.Name.FRONT, \
                  open_dataset.CameraName.Name.FRONT_LEFT, \
                  open_dataset.CameraName.Name.FRONT_RIGHT, \
                  open_dataset.CameraName.Name.SIDE_LEFT, \
                  open_dataset.CameraName.Name.SIDE_RIGHT]:
        image_id += 1
        labeled_image = LabeledImage(image_id)
        labeled_image.daylight = frame.context.stats.time_of_day
        labeled_image.location = frame.context.stats.location
        count_vehicles = 0
        count_pedestrians = 0
        count_signs = 0
        count_cyclists = 0
        total_area_vehicles = 0
        width = 0
        height = 0
        for camera_calibration in frame.context.camera_calibrations:
            if camera_calibration.name != angle:
                continue
            width = camera_calibration.width
            height = camera_calibration.height
        for camera_label in frame.camera_labels: # 2D labels
            if camera_label.name != angle:
                continue
            for label in camera_label.labels:
                got2Dlabels = True
                labeled_image.add_label(label, width, height)
                if label.type == open_labels.Label.Type.TYPE_VEHICLE:
                    count_vehicles += 1
                    total_area_vehicles += label.box.width * label.box.length
                if label.type == open_labels.Label.Type.TYPE_PEDESTRIAN:
                    count_pedestrians += 1
                if label.type == open_labels.Label.Type.TYPE_SIGN:
                    count_signs += 1
                if label.type == open_labels.Label.Type.TYPE_CYCLIST:
                    count_cyclists += 1
        for camera_label in frame.projected_lidar_labels: # Projected label from lidar 3D points
            if camera_label.name != angle:
                continue
            for label in camera_label.labels:
                labeled_image.add_label(label, width, height)
                if label.type == open_labels.Label.Type.TYPE_VEHICLE:
                    count_vehicles += 1
                    total_area_vehicles += label.box.width * label.box.length
                if label.type == open_labels.Label.Type.TYPE_PEDESTRIAN:
                    count_pedestrians += 1
                if label.type == open_labels.Label.Type.TYPE_SIGN:
                    count_signs += 1
                if label.type == open_labels.Label.Type.TYPE_CYCLIST:
                    count_cyclists += 1
        labeled_image.count_vehicles = count_vehicles
        labeled_image.count_pedestrians = count_pedestrians
        labeled_image.count_signs = count_signs
        labeled_image.count_cyclists = count_cyclists
        labeled_image.total_area_vehicles = total_area_vehicles
        for image in frame.images:
            if image.name != angle:
                continue
            labeled_image.image_data = img.open(io.BytesIO(image.image))
        labeled_image.build_paths()
        if len(labeled_image.path_train) > 0 \
                or len(labeled_image.path_test) > 0:
            if len(labeled_image.labels) > 0:
                labeled_image.save()
                labeled_image.add_to_links_files()


def main(argv):
    # Code to partition images based on Generative Factors
    tf.compat.v1.enable_eager_execution()
    global train_folder
    for folder in train_folders:
        train_folder = folder
        train_folder_path = folder_path + train_folder
        global image_id
        image_id = 0
        for root, dirs, files in os.walk(train_folder_path):
            for file in files:
                if file.startswith("segment"):  # avoid license file inside the folder
                    filepath = os.path.join(root, file)
                    print("examine : " + filepath)
                    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
                    for data in dataset:
                        frame = open_dataset.Frame() # instance of a frame
                        frame.ParseFromString(bytearray(data.numpy()))
                        examine_frame(frame)
    tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    main(sys.argv)




