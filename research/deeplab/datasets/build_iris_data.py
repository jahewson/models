# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts Iris data to TFRecord file format with Example protos."""

import io
import math
import os
import random
import sys
import build_data
import tensorflow as tf
import PIL.Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_folder',
    './iris/images/training',
    'Folder containing trainng images')

tf.app.flags.DEFINE_string(
    'val_image_folder',
    './iris/images/validation',
    'Folder containing validation images')

tf.app.flags.DEFINE_string(
    'output_dir', './iris/tfrecord',
    'Path to save converted tfrecord of Tensorflow example')

_NUM_SHARDS = 1


def _convert_dataset(dataset_split, dataset_dir):
  """Converts the Iris dataset into into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val).
    dataset_dir: Dir in which the dataset locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """

  img_names = tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))
  random.shuffle(img_names)

  num_images = len(img_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)

      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d (%s)' % (
            i + 1, num_images, shard_id, dataset_split))
        sys.stdout.flush()

        # Read the image.
        image_filename = img_names[i]

        # workaround: crop image as there is a 1px border
        img = PIL.Image.open(image_filename)
        img = img.crop((1, 1, img.width - 2, img.height - 2))
        image_data = image_array_to_png(np.asarray(img))

        height, width = image_reader.read_image_dims(image_data)
        
        # Read the semantic segmentation annotation.
        basename = os.path.basename(image_filename).split('.')[0]
        seg_filename = os.path.join(dataset_dir, basename + '_watershed_mask.png')

        # workaround: crop image as there is a 1px border
        img_seg = PIL.Image.open(seg_filename)
        img_seg = img_seg.crop((1, 1, img_seg.width - 2, img_seg.height - 2))

        # re-compute mask labels
        seg = np.copy(np.asarray(img_seg))
        
        seg[seg == 4]  = 0 # ground
        seg[seg == 23] = 1 # sky
        seg[seg == 8]  = 2 # cloud
        seg[seg == 33] = 3 # building
        seg[seg == 22] = 4 # water

        seg_data = image_array_to_png(seg)

        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')

        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, img_names[i], height, width, seg_data)
            
        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)
  _convert_dataset('train', FLAGS.train_image_folder)
  _convert_dataset('val', FLAGS.val_image_folder)

def image_array_to_png(array):
  img = PIL.Image.fromarray(array)
  output = io.BytesIO()
  img.save(output, format='PNG')
  return output.getvalue()

if __name__ == '__main__':
  tf.app.run()
