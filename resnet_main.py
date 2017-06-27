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

"""ResNet Train/Eval module.
"""
import time
import sys
import pandas as pd

import cifar10_input
import numpy as np
import resnet_model
import tensorflow as tf
import os
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '/home/tmp/cifar10_data',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '/home/tmp/cifar10_data',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '/home/tmp/ResNet_train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/home/tmp/ResNet_eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_bool('use_fp16', False,
                         'Whether use float16 to train the model.')
tf.app.flags.DEFINE_string('log_root', '/home/tmp/ResNet_train',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('train_steps', 120000,
                            'Number of train steps.')
tf.app.flags.DEFINE_integer('learning_rate', 0.1,
                            'The start learning rate.')
tf.app.flags.DEFINE_integer('eval_freq', 200,
                            'After stpes to eval the data.')
tf.app.flags.DEFINE_integer('report_freq', 20,
                            'After stpes to report the data.')
tf.app.flags.DEFINE_integer('save_checkpoint_freq', 1000,
                            'After stpes to save the checkpoints.')
tf.app.flags.DEFINE_integer('eval_batch_size', 200,
                            'Evaluation batch size.')
tf.app.flags.DEFINE_integer('train_batch_size', 128,
                            'Training batch size.')


def train(hps):
    train_images, train_labels = resnet_model.inputs(FLAGS.train_data_path,
                                                     FLAGS.train_batch_size)
    eval_images, eval_labels = resnet_model.inputs(FLAGS.eval_data_path,
                                                   FLAGS.eval_batch_size, eval_data=True)

    train_model = resnet_model.ResNet(hps, train_images, train_labels, 'train')
    train_model.build_graph()
    eval_model = resnet_model.ResNet(hps, eval_images, eval_labels, 'eval')
    eval_model.build_graph()

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())

    train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    eval_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)

    ckpt = tf.train.get_checkpoint_state(FLAGS.log_root)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from checkpoint')
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(init)
        print('Train the model from scratch')
        global_step = 0

    print('Start training')
    print('----------------------------')
    tf.train.start_queue_runners(sess=sess)

# These lists are used to save a csv file at last
    step_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    eval_step_list = []

    for train_step in range(global_step, FLAGS.train_steps):
        # The learning rate of wide ResNet.
        if train_step < 24000:
            lrn_rate = FLAGS.learning_rate
        elif train_step < 48000:
            lrn_rate = FLAGS.learning_rate/5
        elif train_step < 64000:
            lrn_rate = FLAGS.learning_rate/25
        elif train_step < 90000:
            lrn_rate = FLAGS.learning_rate/125
        else:
            lrn_rate = FLAGS.learning_rate/625

        # The learning rate of ResNet.
        # if train_step < 40000:
        #     lrn_rate = FLAGS.learning_rate
        # elif train_step < 60000:
        #     lrn_rate = FLAGS.learning_rate/10
        # elif train_step < 80000:
        #     lrn_rate = FLAGS.learning_rate/100
        # else:
        #     lrn_rate = FLAGS.learning_rate/1000

        # The learning rate of ResNext.
        # if train_step < 60000:
        #     lrn_rate = FLAGS.learning_rate
        # elif train_step < 90000:
        #     lrn_rate = FLAGS.learning_rate/10
        # else:
        #     lrn_rate = FLAGS.learning_rate/100

        eval_batch_count = 5  if train_step < 40000 else FLAGS.eval_batch_count
        # train step
        start_time = time.time()
        feed_dict = {train_model.lrn_rate: lrn_rate}
        _, lost, acc, lr, summary_str = sess.run([train_model.train_op, train_model.costs,
            train_model.acc, train_model.lrn_rate, train_model.summaries], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(lost), 'Model diverged with loss = NaN'
        # print(train_step)
        # print('lost:', lost)
        # print('acc:', acc)

        # write summary and reprot every report_freq steps
        if train_step % FLAGS.report_freq == 0:
            # summary_str = sess.run([train_model.summaries], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, train_step)

            num_examples_per_step = FLAGS.train_batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
            print(format_str % (datetime.now(), train_step, lost, examples_per_sec,
                sec_per_batch))
            print('Train top1 acc = ', acc)
            # print('Validation top1 error = %.4f' % validation_error_value)
            # print('Train loss = ', lost)
            print('----------------------------')

            train_loss_list.append(lost)
            step_list.append(train_step)
            df = pd.DataFrame(data={'step':step_list, 'train_loss':train_loss_list})
            df.to_csv(FLAGS.train_dir + '_train_loss.csv')

        # evaluate every setted steps
        if (train_step + 1) % FLAGS.eval_freq == 0:
            print('Running evaluation...')
            eval_loss, eval_acc, n_batch = 0, 0, 0
            for i in range(eval_batch_count):
                # eval_batch_images, eval_batch_labels = eval_model.inputs(eval_data=True)
                feed_dict = {eval_model.lrn_rate: lrn_rate}
                _, lost, acc, summary= sess.run([eval_model.eval_op, eval_model.costs,
                    eval_model.acc, eval_model.summaries],feed_dict=feed_dict)
                print(acc,lost)
                eval_writer.add_summary(summary, train_step + i)
                eval_loss += lost
                eval_acc += acc
                n_batch += 1
                # print('eval step', i+1)
                # print('eval loss', lost)
                # print('eval accuracy', acc)
            tot_eval_loss = eval_loss / n_batch
            tot_eval_acc = eval_acc / n_batch
            print('   eval loss: {}'.format(tot_eval_loss))
            print('   eval accuracy: {}'.format(tot_eval_acc))
            print('----------------------------')
            eval_step = train_step + 1
            val_acc_list.append(tot_eval_acc)
            val_loss_list.append(tot_eval_loss)
            eval_step_list.append(eval_step)
            vali_summ = tf.Summary()
            vali_summ.value.add(tag='validation_acc',
                simple_value=tot_eval_acc.astype(np.float))
            train_writer.add_summary(vali_summ, train_step)
            train_writer.flush()

            df2 = pd.DataFrame(data={'step':eval_step_list, 'validation_acc':val_acc_list,
                'validation_loss':val_loss_list})
            df2.to_csv(FLAGS.train_dir + '_eval_acc.csv')

        # save checkpoints
        if train_step % FLAGS.save_checkpoint_freq == 0:
             checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=train_step)

def test(hps):
    images, labels = resnet_model.inputs(
        FLAGS.eval_data_path, FLAGS.eval_batch_size, eval_data=True)
    model = resnet_model.ResNet(hps, images, labels, 'eval')
    model.build_graph()
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # while True:
    #     try:
    #         ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    #     except tf.errors.OutOfRangeError as e:
    #         tf.logging.error('Cannot restore checkpoint: %s', e)
    #     continue
    #     if not (ckpt_state and ckpt_state.model_checkpoint_path):
    #         tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
    #     continue
    ckpt = tf.train.get_checkpoint_state(FLAGS.log_root)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
    else:
        tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
        return

    tf.train.start_queue_runners(sess)
    test_acc, test_loss = 0, 0
    print('start')
    for _ in range(FLAGS.eval_batch_count):
        (summaries, loss, predictions, acc, train_step) = sess.run(
            [model.summaries, model.costs, model.labels, model.acc, model.global_step])
        test_acc += acc
        test_loss += loss
        print(acc)

    precision = 1.0 * test_acc / FLAGS.eval_batch_count
    total_loss = 1.0 * test_loss / FLAGS.eval_batch_count

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f' % (total_loss, precision))
    summary_writer.flush()

def main(_):
    resnet_model.maybe_download_and_extract()
    if FLAGS.num_gpus == 0:
        dev = '/CPU:0'
    elif FLAGS.num_gpus == 1:
        dev = '/GPU:0'
    else:
        raise ValueError('Only support 0 or 1 gpu')

    # if FLAGS.mode == 'train':
    #     batch_size = 128
    # elif FLAGS.mode == 'eval':
    #     batch_size = FLAGS.eval_batch_size

    if FLAGS.dataset == 'cifar10':
        num_class = 10
    elif FLAGS.dataset == 'cifar100':
        num_class = 100

    hps = resnet_model.HParams(num_class=num_class,
                               lrn_rate=0.1,
                               num_residual_units=6,
                               use_bottleneck=False,
                               weight_decay_rate=0.0005,
                               dropout_rate=0.3,
                               relu_leakiness=0.1,
                               optimizer='mom',
                               width=10,
                               data_dir=FLAGS.train_data_path)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            test(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
