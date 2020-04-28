# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

try:
    filename = sys.argv[1]
    
except IndexError:
    print('Usage: python object_detection_image.py <filename>')
    quit()

except KeyError as err:
    frame = inspect.currentframe()
    print('{}:{}: error: Key {} not found in input data file'.format(__file__,frame.f_lineno,err))
    quit()
    
# Name of the directory containing the object detection module we're using
IMAGE_NAME = filename

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
print(image_tensor)
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

width = image.shape[1]
height = image.shape[0]
imageN = np.zeros((width, height, 3), np.uint8)
imageN[:] = (0, 0, 255)
objBBL = []
min_score_thresh=0.60
true_boxes = boxes[0][scores[0] > min_score_thresh]
for i in range(true_boxes.shape[0]):
    ymin = int(true_boxes[i,0]*height)
    xmin = int(true_boxes[i,1]*width)
    ymax = int(true_boxes[i,2]*height)
    xmax = int(true_boxes[i,3]*width)
    objBB = [xmin,ymin, xmax,ymin, xmax,ymax,xmin,ymax]
    objBBL.append(objBB)
    
print(objBBL)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()



