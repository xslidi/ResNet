# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Rotine for decoding the CIFAR-10 binary file format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. We will follow the simple data augmentation for training:
# 4 pixels are padded on each side.
IMAGE_SIZE = 32
Padding = 8

# Global constants describing the CIFAR-10/100 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue, dataset):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.
        dataset: Either 'cifar10' or 'cifar100'

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    class CIFAR_Recorder(object):
        pass
    result = CIFAR_Recorder()

    # Dimensions of the images in the CIFAR-10/100 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    if dataset == 'cifar10':
        label_bytes = 1
        NUM_CLASSES = 10
    elif dataset == 'cifar100':
        label_bytes = 2  # 2 for CIFAR-100
        NUM_CLASSES = 100
    else:
        raise ValueError('Not supported dataset %s', dataset)

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [label_bytes-1], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
                       [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(eval_data, image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    if not eval_data:
        num_preprocess_threads = 16
    else:
        num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    if not eval_data:
        # filenames = tf.gfile.Glob(data_dir)
        if FLAGS.dataset == 'cifar10':
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                            for i in range(1, 6)]
        elif FLAGS.dataset == 'cifar100':
            filenames = [os.path.join(data_dir, 'train.bin')]

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        # for f in filenames:
        #     if not tf.gfile.Exists(f):
        #         raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue, FLAGS.dataset)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        # image data processing
        # 4 pixels are padding on each sides
        distorted_image = tf.image.resize_image_with_crop_or_pad(
            reshaped_image, height + Padding, width + Padding)

        # randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # randomly sampled from the image
        distorted_image = tf.random_crop(distorted_image, [IMAGE_SIZE, IMAGE_SIZE, 3])

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # distorted_image = tf.image.random_brightness(distorted_image,
        #                                         max_delta=63. / 255.)
        # distorted_image = tf.image.random_contrast(distorted_image,
        #                                         lower=0.2, upper=1.8)
        # distorted_image = tf.image.random_saturation(distorted_image,
        #                                         lower=0.5, upper=1.5)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
    else:
        # filenames = tf.gfile.Glob(data_dir)
        if FLAGS.dataset == 'cifar10':
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
        elif FLAGS.dataset == 'cifar100':
            filenames = [os.path.join(data_dir, 'test.bin')]

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue, FLAGS.dataset)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        # image data processing
        # 0 pixels are padding on each sides
        distorted_image = tf.image.resize_image_with_crop_or_pad(
            reshaped_image, height, width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(eval_data, float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
