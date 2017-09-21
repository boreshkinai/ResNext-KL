
from collections import namedtuple

import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import models.architectures as architectures

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, use_hyper, projection_width, architecture')

COLUMN_MAP_NAME='columns.pkl'

class Net(object):



    def __init__(self, flags, hps, mode, session=None):
        if mode=='train':
            self.is_training = True
        else:
            self.is_training = False
        self.mode = mode
        self.hps = hps
        self.flags = flags
        self.sess = session


    def cosine_decay(self, learning_rate, global_step, max_step, name=None):
        from tensorflow.python.framework import ops
        from tensorflow.python.ops import math_ops
        from tensorflow.python.framework import constant_op

        with ops.name_scope(name, "CosineDecay",
                            [learning_rate, global_step, max_step]) as name:

            learning_rate = ops.convert_to_tensor(0.5*learning_rate, name="learning_rate")
            dtype = learning_rate.dtype
            global_step = math_ops.cast(global_step, dtype)

            const = math_ops.cast(constant_op.constant(1), learning_rate.dtype)

            freq = math_ops.cast(constant_op.constant(np.pi/max_step), learning_rate.dtype)
            osc = math_ops.cos(math_ops.multiply(freq, global_step))
            osc = math_ops.add(osc, const)

            return math_ops.multiply(osc, learning_rate, name=name)


    def train_column(self, images, labels):

        logits1 = self.predict_columns(images=images, is_training=True)
        # logits2 = h2[-1]

        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=tf.one_hot(labels, 10)))
        # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=tf.one_hot(labels, 10)))

        regu_losses = slim.losses.get_regularization_losses()
        total_loss = tf.add_n([loss1] + regu_losses) #  + [loss2]

        misclass1 = 1.0 - slim.metrics.accuracy(tf.argmax(logits1, 1), labels)
        tf.summary.scalar('misclassification1', misclass1)
        # misclass2 = 1.0 - slim.metrics.accuracy(tf.argmax(logits2, 1), labels)
        # tf.summary.scalar('misclassification2', misclass2)


        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        learning_rate = tf.train.piecewise_constant(global_step, [np.int64(self.flags.number_of_steps/2), np.int64(3*self.flags.number_of_steps/4)], [0.1/8.0, 0.01/8.0, 0.001/8.0])
        #learning_rate = self.cosine_decay(learning_rate=0.05, global_step=global_step, max_step=self.flags.number_of_steps, name='cosine_decay')

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer, global_step=global_step)

        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)

        logdir = self._get_train_logdir()
        slim.learning.train(
            train_op=train_op,
            logdir=logdir,
            number_of_steps=self.flags.number_of_steps,
            save_summaries_secs=self.flags.save_summaries_secs,
            save_interval_secs=self.flags.save_interval_secs,
            summary_writer=tf.summary.FileWriter(logdir, flush_secs=30),
        )


    def eval_column(self, images, labels):

        logits = self.predict_columns(images=images, is_training=False)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.one_hot(labels, 10)))

        regu_losses = slim.losses.get_regularization_losses()
        total_loss = tf.add_n([loss] + regu_losses)

        accuracy, update = slim.metrics.streaming_accuracy(tf.argmax(logits, 1), labels)
        metric_dictionary = {'misclassification': (1.0 - accuracy, update)}

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_dictionary)

        summary_ops = []
        for metric_name, metric_value in names_to_values.items():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.summary.scalar('loss', total_loss))

        slim.get_or_create_global_step()
        slim.evaluation.evaluation_loop(
            None,
            checkpoint_dir=self._get_train_logdir(),
            logdir=self._get_eval_logdir(),
            num_evals=self.flags.num_evals,
            max_number_of_evaluations=self.flags.max_number_of_evaluations,
            eval_op=names_to_updates,
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=self.flags.eval_interval_secs)


    def _get_train_logdir(self):
        logdir = os.path.join(
            os.path.expanduser(self.flags.log_dir), 'train')
        return logdir


    def _get_eval_logdir(self):
        logdir = os.path.join(
            os.path.expanduser(self.flags.log_dir), 'eval')
        return logdir


    @staticmethod
    def _get_scope(is_training):
        conv2d_arg_scope = slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'epsilon': 0.001,
                'momentum': .9,
                'trainable': is_training,
                'training': is_training,
            },
            padding='SAME',
            trainable=is_training,
            weights_initializer=architectures.ScaledVarianceRandomNormal(),
            biases_initializer=tf.constant_initializer(0.0)
        )
        dropout_arg_scope = slim.arg_scope(
            [slim.dropout],
            keep_prob=0.5,
            is_training=is_training)
        return conv2d_arg_scope, dropout_arg_scope


    def predict_columns(self, images, is_training):
        conv2d_arg_scope, dropout_arg_scope = self._get_scope(is_training)

        h_stack1 = []
        h_stack2 = []
        with conv2d_arg_scope, dropout_arg_scope:
            h = images
            h_stack1.append(h)
            h_stack2.append(h)

            for layer_idx, layer in enumerate(self.hps.architecture[:-1]):
                adaptor = self.hps.adaptors[layer_idx]
                with tf.variable_scope('col1'):
                    h1 = layer.apply(h_stack1[-1])
                    if adaptor != None:
                        h2_1 = adaptor.apply(h_stack2[-1])
                        h1 = tf.add(h1, h2_1, name=layer.scope + '/add_lateral')
                        h1 = tf.nn.relu(h1, name=layer.scope + '/relu')

                with tf.variable_scope('col2'):
                    h2 = layer.apply(h_stack2[-1])
                    if adaptor != None:
                        h1_2 = adaptor.apply(h_stack1[-1])
                        h2 = tf.add(h2, h1_2, name=layer.scope + '/add_lateral')
                        h2 = tf.nn.relu(h2, name=layer.scope + '/relu')

                h_stack1.append(h1)
                h_stack2.append(h2)

            # # process the fully connected layer
            # with tf.variable_scope('col1'):
            #     h1 = self.hps.architecture[-1].apply(h_stack1[-1])
            #     h_stack1.append(h1)
            #     flat_logits1 = slim.flatten(h1, scope='logits/flat_logits')
            #     h_stack1.append(flat_logits1)
            #
            # with tf.variable_scope('col2'):
            #     h2 = self.hps.architecture[-1].apply(h_stack2[-1])
            #     h_stack2.append(h2)
            #     flat_logits2 = slim.flatten(h2, scope='logits/flat_logits')
            #     h_stack2.append(flat_logits2)

            with tf.variable_scope('fuse'):
                lateral_stack = tf.concat(values=[h_stack1[-1], h_stack2[-1]], axis=-1, name='/concat_lateral')
                logits = self.hps.architecture[-1].apply(lateral_stack)
                flat_logits = slim.flatten(logits, scope='logits/flat_logits')

        return flat_logits