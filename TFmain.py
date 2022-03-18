'''
all resources were forked from https://github.com/yatharth-b/CSGO-aimbot and 
https://gist.github.com/imamdigmi/b203d120953db4ef2a89ca7ce70da0a1#file-video_save-py-L8
eddited by RockyLucky

---notes---
if you get TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
its because it didn't find a object for detection
'''
import pathlib
import cv2
import cv2 as cv
import mss
import time
import sys

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import os
import tensorflow_hub as hub

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# PARAMETERS--------------------------------
# video file to run detection 
video = 'test2'

# Playing video from file
cap = cv2.VideoCapture(video +'.MP4')

FILE_OUTPUT = video + '.AVI'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame_width, frame_height))

sys.path.append("..")


# ## Env setup
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # Model preparation 
PATH_TO_FROZEN_GRAPH = 'CSGO_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'CSGO_labelmap.pbtxt'
NUM_CLASSES = 4

# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)


            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              frame,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)

            array_ch = []
            array_c = []
            array_th = []
            array_t = []
            for i,b in enumerate(boxes[0]):
              if classes[0][i] == 2: # ch
                if scores[0][i] >= 0.5:
                  mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                  mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                  array_ch.append([mid_x, mid_y])
                  cv2.circle(frame,(int(mid_x*frame_width),int(mid_y*frame_height)), 3, (0,0,255), -1)
              if classes[0][i] == 1: # c 
                if scores[0][i] >= 0.5:
                  mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                  mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
                  array_c.append([mid_x, mid_y])
                  cv2.circle(frame,(int(mid_x*frame_width),int(mid_y*frame_height)), 3, (50,150,255), -1)
              if classes[0][i] == 4: # th
                if scores[0][i] >= 0.5:
                  mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                  mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                  array_th.append([mid_x, mid_y])
                  cv2.circle(frame,(int(mid_x*frame_width),int(mid_y*frame_height)), 3, (0,0,255), -1)
              if classes[0][i] == 3: # t
                if scores[0][i] >= 0.5:
                  mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                  mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
                  array_t.append([mid_x, mid_y])
                  cv2.circle(frame,(int(mid_x*frame_width),int(mid_y*frame_height)), 3, (50,150,255), -1)
                

            if ret == True:
                # Saves for video
                out.write(frame)

                # Display the resulting frame
                cv2.imshow('Charving Detection', frame)

                # Close window when "Q" button pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
        

