import os
import importlib
import shutil
import sys
import math
import itertools
import io
from array import array
from PIL import Image as img
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.callbacks import CSVLogger
history = CSVLogger('kerasloss.csv', append=True, separator=';')

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# Specify your Generative Factors and their ranges chosen for 'Waymo dataset'
# Daylight ranges
DayLightRange = ['Day', 'Night', 'Dawn/Dusk']
# Location ranges
LocationRange = ['location_phx', 'location_sf', 'location_other']
# Frame Area ranges
FAD_Vehicle = [['FAD_Vehicle == 0'], ['FAD_Vehicle > 0','FAD_Vehicle < 2195'], ['FAD_Vehicle >= 2195']]
# Num vehicles ranges
Frame_Count_TYPE_VEHICLE = [['Frame_Count_TYPE_VEHICLE == 0'], ['Frame_Count_TYPE_VEHICLE > 0', 'Frame_Count_TYPE_VEHICLE < 60'], ['Frame_Count_TYPE_VEHICLE >= 60', 'Frame_Count_TYPE_VEHICLE < 120'], ['Frame_Count_TYPE_VEHICLE >= 120']]
# pedes ranges
Frame_Count_TYPE_PEDESTRIAN = [['Frame_Count_TYPE_PEDESTRIAN == 0'], ['Frame_Count_TYPE_PEDESTRIAN > 0','Frame_Count_TYPE_PEDESTRIAN < 60'], ['Frame_Count_TYPE_PEDESTRIAN >= 60']]

# automated code to generate folder structure for all combinations of generative factor for the above chosen generative factor combinations
folder_count = 0
folder_path = "/home/gballot/NTU/FSPT Yolo/darknet/waymo/"
for m0 in DayLightRange:
    if(m0 != 'Dawn/Dusk'):
        m0_new = m0
        path = folder_path+ m0_new
    else:
        m0_new = 'Dawn_Dusk' # This done to create a folder named 'Dawn_Dusk' because name with 'Dawn/Dusk' creates problem
        path = folder_path+ m0_new
    if not os.path.exists(path):
        os.mkdir(path)
    for m1 in LocationRange:
        path = folder_path+ m0_new+"/"+m1
        if not os.path.exists(path):
            os.mkdir(path)
        for m2 in FAD_Vehicle:
            x = (str(m2[0])+str(m2[1])) if(len(m2) > 1) else str(m2[0])
            path = folder_path+  m0_new+"/"+m1+"/"+x
            if not os.path.exists(path):
                os.mkdir(path)
            for m3 in Frame_Count_TYPE_VEHICLE:
                y = (str(m3[0])+str(m3[1])) if(len(m3) > 1) else str(m3[0])
                path = folder_path+  m0_new+"/"+m1+"/"+x+"/"+y
                if not os.path.exists(path):
                    os.mkdir(path)
                for m4 in Frame_Count_TYPE_PEDESTRIAN:
                    z = (str(m4[0])+str(m4[1])) if(len(m4) > 1) else str(m4[0])
                    path = folder_path+  m0_new+"/"+m1+"/"+x+"/"+y+"/"+z
                    if not os.path.exists(path):
                        os.mkdir(path)
                    folder_count=folder_count+1
print(folder_count)


train_folder ='training_0000'
train_folder_path = folder_path + train_folder
output_file = open(folder_path + 'generated_code.py', 'w')
stdout = sys.stdout
sys.stdout = output_file


head = """# This code has been automaticaly generated
import os
import importlib
import shutil
import tensorflow as tf
import math
import itertools
import io
from array import array
from PIL import Image as img
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.callbacks import CSVLogger
history = CSVLogger('kerasloss.csv', append=True, separator=';')

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
tf.compat.v1.enable_eager_execution()

# Specify your Generative Factors and their ranges chosen for 'Waymo dataset'
# Daylight ranges
DayLightRange = ['Day', 'Night', 'Dawn/Dusk']
# Location ranges
LocationRange = ['location_phx', 'location_sf', 'location_other']
# Frame Area ranges
FAD_Vehicle = [['FAD_Vehicle == 0'], ['FAD_Vehicle > 0','FAD_Vehicle < 2195'], ['FAD_Vehicle >= 2195']]
# Num vehicles ranges
Frame_Count_TYPE_VEHICLE = [['Frame_Count_TYPE_VEHICLE == 0'], ['Frame_Count_TYPE_VEHICLE > 0', 'Frame_Count_TYPE_VEHICLE < 60'], ['Frame_Count_TYPE_VEHICLE >= 60', 'Frame_Count_TYPE_VEHICLE < 120'], ['Frame_Count_TYPE_VEHICLE >= 120']]
# pedes ranges
Frame_Count_TYPE_PEDESTRIAN = [['Frame_Count_TYPE_PEDESTRIAN == 0'], ['Frame_Count_TYPE_PEDESTRIAN > 0','Frame_Count_TYPE_PEDESTRIAN < 60'], ['Frame_Count_TYPE_PEDESTRIAN >= 60']]

folder_path = '""" + folder_path + """'
# Code to partition images based on Generative Factors
Sum_Average_Frame_Area_TYPE_VEHICLE = 0
Total_Average_Frame_Area_TYPE_VEHICLE = 0
Total_AREA_TYPE_VEHICLE = 0 # Sum of area occupied by vehicles in a frame
FAD_Threshold = 2000 # Threshold value for sum of area occupied by vehicles in a frame
Num_Veh_Threshold = 30 # Threshold number of vehicles in a frame
Frame_Count_TYPE_PEDESTRIAN_Threshold = 5 # number of pedestrain in an image
image_count = 0
train_folder = '""" + train_folder + """'
train_folder_path = '""" + train_folder_path + """'
for root, dirs, files in os.walk(train_folder_path):
    for file in files:
        if file.startswith("segment"):  # avoid license file inside the folder
            filepath = os.path.join(root, file)
            #print(filepath)
            dataset = tf.data.TFRecordDataset(filepath, compression_type='')
            for data in dataset:
                frame = open_dataset.Frame() # instance of a frame
                frame.ParseFromString(bytearray(data.numpy()))
                if not(len(frame.laser_labels)==0):
                 # do this for every frame -
                    Frame_Count_TYPE_PEDESTRIAN = 0
                    Frame_Count_TYPE_VEHICLE = 0
                    FAD_Vehicle = 0
                    for m in range(0,len(frame.laser_labels)): # selects  frame's front camera and label 'm'
                        if(frame.laser_labels[m].type == 2 or frame.laser_labels[m].type == 4):  # Type pedestrian and cyclist
                            Frame_Count_TYPE_PEDESTRIAN =Frame_Count_TYPE_PEDESTRIAN+1
                        if(frame.laser_labels[m].type == 1):  # Type vehicle
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
                            """



print(head)

#######################################
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
                        print("                                                imag.save(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+train_folder+str(image_count)+'.jpg')" %(m0,m1,x,y,z))
                        print("                                                print(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+train_folder+str(image_count)+'.jpg')" %(m0,m1,x,y,z))
                    else:
                        m0_new = 'Dawn_Dusk'
                        print("                                                imag.save(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+train_folder+str(image_count)+'.jpg')" %(m0_new,m1,x,y,z))
                        print("                                                print(folder_path+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+'%s'+'/'+train_folder+str(image_count)+'.jpg')" %(m0_new,m1,x,y,z))
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
