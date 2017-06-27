# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import tensorflow as tf
import numpy as np
import os
import tarfile
import urllib
import sys

from collections import namedtuple
from tensorflow.python.training import moving_averages

import cifar10_input

FLAGS = tf.app.flags.FLAGS
DATA_URL_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_URL_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
# Basic model parameters.
HParams = namedtuple('HParams',
                     'num_class, lrn_rate, num_residual_units,'
                     'use_bottleneck, weight_decay_rate, dropout_rate,'
                     'relu_leakiness, optimizer, width, data_dir')

Momentum = 0.9
epslion = 0.001

class ResNet(object):
    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.
        Args:
            hps: Hyperparameters.
            images: Batches of images. [batch_size, image_size, image_size, 3]
            labels: Batches of labels. [batch_size, num_classes]
            mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self.images = images
        self.labels = labels
        self.mode = mode
        self._extra_train_ops = []

    # Denefine the basic function of the ResNet
    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _variable_with_weight_decay(self, name, shape, initializer, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            shape: list of ints
            initializer: the initializer that variable used
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

        Returns:
            Variable Tensor
        """
            #   dtype = tf.float16 if self.hps.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _batch_norm_layer(self, name, input_layer):
        '''Batch normlization layer
            use the running mean and running variance at the test-time.
        '''
        with tf.variable_scope(name):
            params_shape = [input_layer.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape,
                tf.float32, initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            if self.mode == 'train':
                mean, variance = tf.nn.moments(input_layer, [0, 1, 2], name='moments')

                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, Momentum)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, Momentum)
                self._extra_train_ops = [update_moving_mean, update_moving_variance]
            else:
                mean, variance = moving_mean, moving_variance
                self._activation_summary(mean)
                self._activation_summary(variance)
            with tf.control_dependencies(self._extra_train_ops):
                y = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, epslion)
                y.set_shape(input_layer.get_shape())
            return y

    def _conv_layer(self, name, input_layer, filter_size, in_filters, out_filters,
        strides):
        """Convlutional layer

            Args:
                filter_size: the size of eace filter, only consider the square filter.
                in_filters: the number of filters from the last layer.
                out_filters: the number of filters in this layer.
                strides: a 4D tensor, the strides for each filter in convlutional layer,
                    its shape is like [1, stride, stride, 1].
            Returns:
                a tensorm, with the format like [N, H, W, C]
        """
        with tf.variable_scope(name):
            n = filter_size * filter_size * in_filters
            kernal = self._variable_with_weight_decay('weights',
                [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0/n/(1+np.square(self.hps.relu_leakiness)))),
                wd=self.hps.weight_decay_rate)
            return tf.nn.conv2d(input_layer, kernal, strides, padding='SAME')

    def _leaky_relu(self, input_layer, leakness=0.0):
        '''Define Leaky ReLU layer.
           The formulation: f(x) = max(a*x, x), if a = 0, this is a common ReLU.

           Args:
                leakness: the decay parameter multiple to inputs.
        '''
        return tf.where(tf.less(input_layer, 0.0), leakness * input_layer,
            input_layer, name='leaky_ReLU')

    def _global_avg_pool(self, input_layer):
        '''Define the average pooling function'''
        assert input_layer.get_shape().ndims == 4
        return tf.reduce_mean(input_layer, [1, 2])

    def _fully_connected_layer(self, input_layer, out_dim):
        '''Build fully connected layer'''
        if self.mode == 'train':
            x = tf.reshape(input_layer, [FLAGS.train_batch_size, -1])
        else:
            x = tf.reshape(input_layer, [FLAGS.eval_batch_size, -1])
        w = self._variable_with_weight_decay(
            'weight', [x.get_shape()[-1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
            wd=self.hps.weight_decay_rate)
        b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(x, w, b)

    def _residual_block(self, input_layer, in_filters, out_filters, stride,
        active_before_residual=False, use_prediction_shortcuts=True):
        '''Define a residual block with 2 layers in ResNet.

           Note we only use the latest verion of ResNet, aka pre_activation.
           The structure is bn-relu-conv. This structure is recommended when
           the depth is lower than 100 layers.

           Args:
                in_filters: The filters of last layer.
                out_filters: The number of filters in this conv layer.
                stride: A scalar. The first conv layer stride. it will reshape as the shape
                    like [1, stride, stride, 1]
                use_prediction_shortcuts: A bool, Whether to use predictionshortcuts when
                    the input dimensions don't fit output dimensions. If False, we will use
                    zerp padding to increase the dimensions The detail is in the first
                    paper. We prefer to set False in this block.
                active_before_residual: A bool, if true, we active the input before the
                    residual block. Note that in wide ResNet, you must set this section as
                    True.
           Retures:
                A 4D tensor
        '''
        filter_size = 3
        strides = [1, stride, stride, 1]
        if active_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm_layer('init_bn', input_layer)
                x = self._leaky_relu(x, self.hps.relu_leakiness)
                original_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                original_x = input_layer
                x = self._batch_norm_layer('init_bn', input_layer)
                x = self._leaky_relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sublayer1'):
            x = self._conv_layer('conv1', x, filter_size, in_filters, out_filters,
                    strides)
            # print(x.shape)

        with tf.variable_scope('sublayer2'):
            x = self._batch_norm_layer('bn2', x)
            x = self._leaky_relu(x, self.hps.relu_leakiness)

            if (self.mode == 'train') and (self.hps.dropout_rate > 0):
                with tf.variable_scope('droplayer'):
                    x = tf.nn.dropout(x, 1-self.hps.dropout_rate)

            x = self._conv_layer('conv2', x, filter_size, out_filters, out_filters,
                    [1, 1, 1, 1])
        # choose the mode you used.
        with tf.variable_scope('add'):
            if in_filters != out_filters:
                if not use_prediction_shortcuts:
                    padding_size = (out_filters - in_filters)//2
                    original_x = tf.nn.avg_pool(original_x, strides, strides, padding='VALID')
                    original_x = tf.pad(
                        original_x, [[0,0], [0,0], [0,0], [padding_size, padding_size]],
                        'CONSTANT')
                else:
                    original_x = self._conv_layer('projection_conv', original_x, 1,
                        in_filters, out_filters, strides)
            x += original_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, input_layer, in_filters, out_filters, stride,
        active_before_residual=False, use_prediction_shortcuts=True):
        '''Define a bottleneck residual block with 3 layers in ResNet.

           Note: we only use the latest version of ResNet, aka pre_activation V3
           version. The structure is bn-relu-conv. We prefer this structure when
           the depth is more than 100 layers.

           Args:
                in_filters: The filters of last layer.
                out_filters: The number of filters in this conv layer.
                stride:  A scalar. The first conv layer stride. it will reshape as the
                    shape like [1, stride, stride, 1].
                use_prediction_shortcuts: A bool, Whether to use predictionshortcuts when
                    the input dimensions don't fit output dimensions. If False, we will
                    use zerp padding to increase the dimensions The detail is in the
                    first paper. We prefer to set True in this block.
                active_before_residual: A bool, if true, we active the input before the
                    residual block. Note that in wide ResNet, you must set this section as
                    True.
           Retures:
                A 4D tensor
        '''
        filter_size = 3
        strides = [1, stride, stride, 1]
        if active_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm_layer('init_bn', input_layer)
                x = self._leaky_relu(x, self.hps.relu_leakiness)
                original_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                original_x = input_layer
                x = self._batch_norm_layer('init_bn', input_layer)
                x = self._leaky_relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sublayer1'):
            x = self._conv_layer('conv1', x, 1, in_filters, out_filters/4, strides)

        with tf.variable_scope('sublayer2'):
            x = self._batch_norm_layer('bn2', x)
            x = self._leaky_relu(x, self.hps.relu_leakiness)

            if (self.mode == 'train') and (self.hps.dropout_rate > 0):
                with tf.variable_scope('droplayer'):
                    x = tf.nn.dropout(x, 1-self.hps.dropout_rate)
            
            x = self._conv_layer('conv2', x, filter_size, out_filters/4, out_filters/4,
                    [1, 1, 1, 1])

        with tf.variable_scope('sublayer3'):
            x = self._batch_norm_layer('bn3', x)
            x = self._leaky_relu(x, self.hps.relu_leakiness)
            x = self._conv_layer('conv3', x, 1, out_filters/4, out_filters,
                    [1, 1, 1, 1])
        # choose the mode you used.
        with tf.variable_scope('add'):
            if in_filters != out_filters:
                if not use_prediction_shortcuts:
                    padding_size = (out_filters - in_filters)//2
                    original_x = tf.nn.avg_pool(original_x, strides, strides,
                        padding='VALID')
                    original_x = tf.pad(
                        original_x, [[0,0], [0,0], [0,0], [padding_size, padding_size]],
                        'CONSTANT')
                else:
                    original_x = self._conv_layer('projection_conv', original_x, 1,
                        in_filters, out_filters, strides)
            x += original_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    # Define the graph
    def _build_model(self, reuse=False):
        '''Build the ResNet model within the graph'''
        with tf.variable_scope('init', reuse=reuse):
            x = self.images
            x = self._conv_layer('conv_init', x, 3, 3, 64, [1, 1, 1, 1])

        stride = [1, 2, 2]
        # In ResNet.
        preactivate_label = [True, False, False]
        # You should set all preactivate_label to True when you want to use wide ResNet.
        # preactivate_label = [True, True, True]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [64, 64*self.hps.width, 128*self.hps.width, 256*self.hps.width]
        else:
            res_func = self._residual_block
            filters = [64, 16*self.hps.width, 32*self.hps.width, 64*self.hps.width]

        with tf.variable_scope('Unit_1_0', reuse=reuse):
            x = res_func(x, filters[0], filters[1], stride[0],
                active_before_residual=preactivate_label[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('Unit_1_%d' %i, reuse=reuse):
                x = res_func(x, filters[1], filters[1], stride[0])

        with tf.variable_scope('Unit_2_0', reuse=reuse):
            # print(filters[1],filters[2],stride[1])
            x = res_func(x, filters[1], filters[2], stride[1],
                active_before_residual=preactivate_label[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('Unit_2_%d' %i, reuse=reuse):
                x = res_func(x, filters[2], filters[2], stride[0])

        with tf.variable_scope('Unit_3_0', reuse=reuse):
            x = res_func(x, filters[2], filters[3], stride[2],
                active_before_residual=preactivate_label[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('Unit_3_%d' %i, reuse=reuse):
                x = res_func(x, filters[3], filters[3], stride[0])

        with tf.variable_scope('Unit_last', reuse=reuse) :
            x = self._batch_norm_layer('final_bn', x)
            x = self._leaky_relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logits', reuse=reuse):
            x_fc = self._fully_connected_layer(x, self.hps.num_class)
            self.prediction = tf.nn.softmax(x_fc)

        with tf.variable_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=x_fc)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy_mean')
            tf.add_to_collection('losses', cross_entropy_mean)
            self.costs = tf.add_n(tf.get_collection('losses'), name='total_loss')

            tf.summary.scalar('total_loss',self.costs)
        # Calculate the accuarcy of one batch.
        with tf.variable_scope('accuracy'):
            # predictions = tf.cast(tf.argmax(self.prediction, axis=1), tf.int32)
            # correct_prediction = tf.to_float(tf.equal(prediction, self.labels))
            # self.acc = tf.reduce_mean(correct_prediction, axis=1, name='bathc_accuracy')
            correct_prediction = tf.to_float(tf.nn.in_top_k(self.prediction, self.labels, k=1))
            self.acc = tf.reduce_mean(correct_prediction, name='batch_accuracy')

            tf.summary.scalar('batch_accuracy', self.acc)

    def build_graph(self):
        '''Build the training graph'''
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        if self.mode == 'train':
            self._build_model()
            self._build_train_op()
        elif self.mode == 'eval':
            self._build_model(reuse=True)
            self._build_eval_op()
        self.summaries = tf.summary.merge_all()

    def _build_train_op(self):
        '''Build the training option for the model'''
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

        grads = optimizer.compute_gradients(self.costs)
        apply_op = optimizer.apply_gradients(grads, global_step=self.global_step,
            name='train_step')

        # Compute the moving average of all individual losses and the total loss.
        moving_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        train_averages_op = moving_averages.apply([self.costs, self.acc])

        tf.summary.scalar('train_loss_avg', moving_averages.average(self.costs))
        tf.summary.scalar('train_acc_avg', moving_averages.average(self.acc))

        train_ops = [apply_op] + [train_averages_op]
        self.train_op = tf.group(*train_ops)

    def _build_eval_op(self):
        '''Build the evaluation option for the model'''
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        # Compute the moving average of all individual losses and the total loss.
        moving_average = tf.train.ExponentialMovingAverage(0.9, name='avg')
        eval_averages_op = moving_average.apply([self.costs, self.acc])

        tf.summary.scalar('eval_loss_avg', moving_average.average(self.costs))
        tf.summary.scalar('eval_acc_avg', moving_average.average(self.acc))

        self.eval_op = eval_averages_op

    # def _build_train_eval_op(self):
    #     '''Build the training and evaluation option for the model'''
    #     self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    #     tf.summary.scalar('learning_rate', self.lrn_rate)
    #
    #     if self.hps.optimizer == 'sgd':
    #         optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    #     elif self.hps.optimizer == 'mom':
    #         optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    #
    #     grads = optimizer.compute_gradients(self.costs)
    #     apply_op = optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
    #
    #     # Compute the moving average of all individual losses and the total loss.
    #     moving_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #     train_averages_op = moving_averages.apply([self.costs, self.acc])
    #
    #     tf.summary.scalar('train_loss_avg', moving_averages.average(self.costs))
    #     tf.summary.scalar('train_acc_avg', moving_averages.average(self.acc))
    #
    #     train_ops = [apply_op] + [train_averages_op]
    #     self.train_op = tf.group(*train_ops)
    #
    #     tf.summary.scalar('eval_loss_avg', moving_averages.average(self.costs))
    #     tf.summary.scalar('eval_acc_avg', moving_averages.average(self.acc))
    #
    #     self.eval_op = eval_averages_op

def inputs(data_dir, batch_size, eval_data=False):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    if FLAGS.dataset == 'cifar10':
        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    elif FLAGS.dataset == 'cifar100':
        data_dir = os.path.join(data_dir, 'cifar-100-binary')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.train_data_path
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    if FLAGS.dataset == 'cifar10':
        filename = DATA_URL_10.split('/')[-1]
    elif FLAGS.dataset == 'cifar100':
        filename = DATA_URL_100.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
