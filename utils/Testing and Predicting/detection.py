import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_CONF_THRESH = .6
PATH_TO_SAVED_MODEL = '[CONFIGURE YOUR PATH/saved_model]'
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
PATH_TO_LABELS = '[CONFIGURE YOUR PATH/label_map.pbtxt]'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
IMAGE_PATH = '[CONFIGURE YOUR PATH]'


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


image_np = load_image_into_numpy_array(IMAGE_PATH)
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

# Convert to numpy arrays, and take index [0] to remove the batch dimension.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
# print(detections['detection_classes'])
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=400,
    min_score_thresh=MIN_CONF_THRESH,
    agnostic_mode=False)

plt.figure(figsize=(12, 16))
plt.imshow(image_np_with_detections)
plt.savefig("demo.jpg")
