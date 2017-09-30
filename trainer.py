"""ResNext CIFAR10 classifier."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import datasets
from models import architectures
from models import independent_experts

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('data_dir', './datasets/', 'Path to the data.')
flags.DEFINE_integer('num_samples_train', 50000, 'Number of train samples?')
flags.DEFINE_integer('train_batch_size', 16, 'Training batch size.')
flags.DEFINE_integer('learning_rate', 0.1/8.0, 'Training batch size.')
flags.DEFINE_string('log_dir', '', 'Base directory for logging.')
flags.DEFINE_string('checkpoint_dir', './logs', 'Base directory for checkpoints.')
flags.DEFINE_integer('number_of_steps', int(960000/2), 'Number of steps to train a column')
flags.DEFINE_integer('save_summaries_secs', 300, 'Time between savnum_samples_evaling summaries')
flags.DEFINE_integer('save_interval_secs', 300, 'Time between saving model?')
flags.DEFINE_integer('eval_interval_secs', 0, 'Time between evaluating model?')

flags.DEFINE_integer('cardinality', 2, 'Cardinality of the ResNext model (number of splits in the bottleneck)')

flags.DEFINE_float('max_number_of_evaluations', float('inf'), 'Max number of evaluations to run')
flags.DEFINE_integer('num_samples_eval', 10000, 'Number of evaluation samples?')
flags.DEFINE_integer('eval_batch_size', 50, 'Evaluation batch size?')
flags.DEFINE_integer('num_evals', flags.FLAGS.num_samples_eval / flags.FLAGS.eval_batch_size, 'Number of evaluations in the evaluation phase')

flags.DEFINE_string('mode', 'train', "Execution mode in {'train', 'eval'}.")
flags.DEFINE_string('dataset', 'cifar10_by_class', "Dataset in {'cifar10'}.")
flags.DEFINE_string('file_pattern', 'train-c-0[0-9]-s-000[0-4][0-9]-of-00005', 'Regex for the files used in train/eval')

FLAGS = flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

hps=independent_experts.HParams
hps.architecture = architectures.ArchitectureResNext
hps.adaptors = architectures.AdaptorStack


def get_logdir_name():
    epochs=FLAGS.number_of_steps*FLAGS.train_batch_size/FLAGS.num_samples_train
    if FLAGS.log_dir == '':
        logdir = './logs/' + '-'.join([FLAGS.dataset, 'cardinality', str(FLAGS.cardinality), 'batch', str(FLAGS.train_batch_size), 'lr', str(FLAGS.learning_rate), 'epochs', str(epochs)])
    else:
        logdir = FLAGS.log_dir
    return logdir


def train():
    with tf.Graph().as_default():
        images_cls, labels_cls = datasets.load_batch(
            'cifar10', 'train*', FLAGS.num_samples_train, FLAGS.train_batch_size,
            FLAGS.data_dir + FLAGS.dataset, shuffle=True, augment=True)

        images_mi, labels_mi = datasets.load_batch(
            'cifar10_by_class', 'train*', FLAGS.num_samples_train, FLAGS.train_batch_size,
            FLAGS.data_dir + FLAGS.dataset, shuffle=False, augment=True, common_queue_capacity=FLAGS.train_batch_size, common_queue_min=0, num_readers=1)

        # tf.summary.image('images', images, 16)
        # tf.summary.tensor_summary('labels', labels)

        net = independent_experts.Net(flags=FLAGS, hps=hps, mode=FLAGS.mode, session=sess)

        net.train_column(images=[images_cls,images_mi], labels=[labels_cls,labels_mi])


def evaluate():
    with tf.Graph().as_default():
        images, labels = datasets.load_batch(
            'cifar10', 'train*', FLAGS.num_samples_eval, FLAGS.eval_batch_size,
            FLAGS.data_dir+FLAGS.dataset, shuffle=False, augment=False)

        net = independent_experts.Net(flags=FLAGS, hps=hps, mode=FLAGS.mode, session=sess)

        net.eval_column(images=images, labels=labels)



def main(argv=None):
    if FLAGS.mode not in ('train', 'eval'):
        raise ValueError("'mode' should be in {'train', 'eval'}.")
    if FLAGS.mode == 'train':
        train()
    else:
        evaluate()

if __name__ == '__main__':
    tf.app.run()
