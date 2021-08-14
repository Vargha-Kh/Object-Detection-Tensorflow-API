import hashlib
import io
import logging
import os
import random
import re
from collections import defaultdict
import contextlib2
import pandas as pd
from PIL import Image
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', './train', 'data set directory')
flags.DEFINE_string('output_dir', 'data', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', './label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('csv_file', './train_labels.csv', 'information of plates')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
    match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
    return match.groups()[0]


def dict_to_tf_example(
        file_name,
        data,
        label_map_dict,
        image_subdirectory):
    img_path = os.path.join(image_subdirectory, file_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width, height = image.size

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    for obj in data:
        # xmin = float(obj['xmin'])
        # xmax = float(obj['xmax'])
        # ymin = float(obj['ymin'])
        # ymax = float(obj['ymax'])

        # xmins.append(xmin / width)
        # ymins.append(ymin / height)
        # xmaxs.append(xmax / width)
        # ymaxs.append(ymax / height)
        xmn = obj['xmin'] / width
        if xmn < 0.0:
            xmn = 0.0
        elif xmn > 1.0:
            xmn = 1.0
        xmins.append(xmn)

        xmx = obj['xmax'] / width
        if xmx < 0.0:
            xmx = 0.0
        elif xmx > 1.0:
            xmx = 1.0
        xmaxs.append(xmx)

        ymn = obj['ymin'] / height
        if ymn < 0.0:
            ymn = 0.0
        elif ymn > 1.0:
            ymn = 1.0
        ymins.append(ymn)

        ymx = obj['ymax'] / height
        if ymx < 0.0:
            ymx = 0.0
        elif ymx > 1.0:
            ymx = 1.0
        ymaxs.append(ymx)

        class_name = obj['label']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(file_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     image_dir,
                     examples,
                     data
                     ):

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filename, num_shards)
        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))

            # xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
            # mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

            # if not os.path.exists(xml_path):
            #     logging.warning('Could not find %s, ignoring example.', xml_path)
            #     continue
            # with tf.gfile.GFile(xml_path, 'r') as fid:
            #     xml_str = fid.read()
            # xml = etree.fromstring(xml_str)
            # data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            file_name = example
            tf_example = dict_to_tf_example(
                file_name,
                data[file_name],
                label_map_dict,
                image_dir)
            if tf_example:
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Pet dataset.')
    image_dir = data_dir
    df = pd.read_csv(FLAGS.csv_file)
    data = defaultdict(list)
    for row in df.values:
        image, xmin, ymin, xmax, ymax, label = row[:6]
        image = image[:-4] + ".jpg"
        data[image].append(dict(xmin=xmin,
                                ymin=ymin,
                                xmax=xmax,
                                ymax=ymax,
                                label=label))
    examples_list = list(data.keys())
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    create_tf_record(
        train_output_path,
        FLAGS.num_shards,
        label_map_dict,

        image_dir,
        train_examples,
        data)
    create_tf_record(
        val_output_path,
        FLAGS.num_shards,
        label_map_dict,
        image_dir,
        val_examples,
        data)


if __name__ == '__main__':
    tf.app.run()
