import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import math_ops

from . import cifar10
from . import cifar10_by_class
from . import cifar100

_DATASETS = {'cifar10': cifar10,
             'cifar10_by_class': cifar10_by_class,
             'cifar100': cifar100}


def load_example(dataset_name, file_pattern, num_examples, dataset_dir,
                 shuffle=True):
    dataset = _DATASETS[dataset_name].get_split(
        file_pattern, num_examples, dataset_dir)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, shuffle=shuffle, common_queue_capacity=32,
        common_queue_min=0, num_readers=1)
    image, label = data_provider.get(['image', 'label'])

    image = math_ops.cast(image, tf.float32)
    mean = tf.constant(np.asarray([125.3, 123.0, 113.9]).reshape([1, 1, 3]), dtype=tf.float32, name='image_mean')
    std = tf.constant(np.asarray([63.0, 62.1, 66.7]).reshape([1, 1, 3]), dtype=tf.float32, name='image_std')

    whitened_image = tf.div(tf.subtract(image, mean), std)

    return whitened_image, label


def image_augment(image):
    # image = tf.image.per_image_standardization(image)

    transformed_image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    transformed_image = tf.image.random_flip_left_right(transformed_image)
    transformed_image = tf.random_crop(transformed_image, [32, 32, 3])

    return transformed_image


def load_batch(dataset_name, file_pattern, num_examples, batch_size,
               dataset_dir, shuffle=True, augment=True):
    image, label = load_example(dataset_name, file_pattern, num_examples,
                                dataset_dir, shuffle)
    if augment:
        image = image_augment(image)

    return tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=8,
        capacity=32 * batch_size)


def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
    if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
        raise ValueError('Grid shape incompatible with minibatch size.')
    if len(input_tensor.get_shape()) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.get_shape()[1]) != num_features:
            raise ValueError('Image shape and number of channels incompatible '
                             'with input tensor.')
    elif len(input_tensor.get_shape()) == 4:
        incompatible_shape = (
            int(input_tensor.get_shape()[1]) != image_shape[0] or
            int(input_tensor.get_shape()[2]) != image_shape[1] or
            int(input_tensor.get_shape()[3]) != num_channels)
        if incompatible_shape:
            raise ValueError('Image shape and number of channels incompatible '
                             'with input tensor.')
    else:
        raise ValueError('Unrecognized input tensor format.')
    height = grid_shape[0] * image_shape[0]
    width = grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(
        input_tensor, grid_shape + image_shape + [num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(
        input_tensor, [1, height, width, num_channels])
    return input_tensor
