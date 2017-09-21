# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the CIFAR100 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import dataset_utils

TRAIN_FILE_PATTERN = 'train-*-of-*'
TRAIN_NUM_EXAMPLES = 50000
TEST_FILE_PATTERN = 'test-*-of-*'
TEST_NUM_EXAMPLES = 10000


def get_split(file_pattern, num_examples, dataset_dir):
    """Gets a dataset tuple with instructions for reading CIFAR100.

    Parameters
    ----------
    file_pattern : str or sequence of str
        A globable file pattern or a sequence of globable file patterns.
    num_examples : int
        Number of examples spanned by `file_pattern`.
    dataset_dir : str
        The base directory of the dataset sources.

    Returns
    -------
    A `Dataset` namedtuple.

    """

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    if type(file_pattern) in (list, tuple):
        data_sources = [os.path.join(dataset_dir, pattern)
                        for pattern in file_pattern]
    else:
        data_sources = os.path.join(dataset_dir, file_pattern)

    return slim.dataset.Dataset(
        data_sources=data_sources,
        reader=tf.TFRecordReader,
        decoder=slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features={
                'image/encoded': tf.FixedLenFeature((), tf.string, ''),
                'image/format': tf.FixedLenFeature((), tf.string, 'png'),
                'image/class/label': tf.FixedLenFeature(
                    [], tf.int64, tf.zeros([], dtype=tf.int64))},
            items_to_handlers={
                'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
                'label': slim.tfexample_decoder.Tensor('image/class/label'),
            }),
        num_samples=num_examples,
        items_to_descriptions={
            'image': 'A [32 x 32 x 3] color image.',
            'label': 'A single integer between 0 and 99',
        },
        num_classes=10,
        labels_to_names=labels_to_names)
