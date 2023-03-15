#############################################
# SMOKE DETECTOR V1.0 "SMOKY"   NOT USED IN THIS PROJECT
#
# Author: Daniel Eldan R.
# Date  : 05-12-2022
# Mail  : deldanr@gmail.com
# Name  : Detector
# Desc  : Tensorflow based script to load a frozen graph to inference a single image or a set of
############################################

#
# IMPORT BASE LIBRARIES
#
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO
from tinydb import TinyDB, Query
from tinydb.operations import increment
import datetime
#
# IMPORT THE TENSORFLOW OBJECT DETECTION API LIBRARIES
#
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#
# IMPORT LOCAL FUNCTIONS TO SEND A TELEGRAM MSG AND POST A TWEET
#
import telegram_send
import tweet as tw

#
# GLOBAL VARIABLES
#

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Make code runing log more clean
PATH_TO_FROZEN_GRAPH = 'model/smoky5.pb'  # Load our last model: "Smoky"
PATH_TO_LABELS = 'model/label_map.pbtxt'  # Load our label map
TEST_IMAGE_PATHS = [os.path.join('static/test/', '{}'.format(i))  # Load of the test image folder
                    for i in os.listdir('static/test/')]
IMAGE_SIZE = (12, 8)  # A image size for better inference

db = TinyDB("static/data.json")
Todo = Query()

#
# Load the label map
#
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

#
# Funcion to load the model. It return the detection graph
#


def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.graph_util.import_graph_def(od_graph_def, name='')

    return detection_graph


#
# Load the model to background
#
detection_graph = load_model()

#
# Load image and get the shape into an array
#


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#
# Function to run inference on a single image and return a dict with the result
#


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:

            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}

            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph(
                    ).get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:

                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0],
                                             tf.int32)

                detection_boxes = tf.slice(detection_boxes, [0, 0],
                                           [real_num_detection, -1])

                detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                           [real_num_detection, -1, -1])

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])

                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={
                                   image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])

            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
                np.uint8)

            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    db.update(increment('Procesadas'), Todo.Procesadas.exists())

    return output_dict

#
# Function to run a inference on a single image and save the result image file
#


def predict_image(image_path):

    # Open the image file
    image = Image.open(image_path)

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        # IMPORTANT: Here we stablish a 30% of score to draw the inference boxes
        min_score_thresh=.30,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)

    # Save output image
    Image.fromarray(image_np).convert('RGB').save(image_path)

    publish_result(output_dict["detection_scores"],
                   os.path.abspath(image_path))

#
# Run inference on a list of images from a directory. Then, run a function to publish results in Social Network
#


def detection_plot():
    for image_path in TEST_IMAGE_PATHS:

        try:
            image = Image.open(image_path)
        except:
            print("Error abriendo imagen en detection_plot")

        image_np = load_image_into_numpy_array(image)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=0.30,  # 30% of score to draw a box
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)

        # Save the image result and overwrite the input one.
        Image.fromarray(image_np).convert('RGB').save(image_path)

        publish_result(output_dict["detection_scores"],
                       os.path.abspath(image_path))


def publish_result(path):
    
    x = datetime.datetime.now()

    print("********************INCENDIO DETECTADO... PUBLICANDO")

      # Send then Telegram message
    telegram_send.send(messages=["INCENDIO FORESTAL DETECTADO"])
    telegram_send.send(message=f"{x.day}-{x.month}-{x.year} | {x.hour}:{x.minute}:{x.second}")

    with open(path, "rb") as f:
        telegram_send.send(images=[f])
        print("TELEGRAM OK")

        # Example image manipulation
    path2 = os.path.abspath(path)
    img = Image.open(path2)

        # Save image in-memory
    b = BytesIO()
    img.save(b, "PNG")
    b.seek(0)

        # Upload media to Twitter APIv1.1
    ret = tw.api.media_upload(filename="dummy_string", file=b)
    string_tw = "INCENDIO FORESTAL DETECTADO\n" + f"{x.day}-{x.month}-{x.year} | {x.hour}:{x.minute}:{x.second}"
        # Attach media to tweet
    tw.api.update_status(
        media_ids=[ret.media_id_string], status=string_tw)
    print("TWITTEADO OK")

