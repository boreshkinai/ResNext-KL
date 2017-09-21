import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import init_ops
import numpy as np

WEIGHT_DECAY = 0.0005

class ScaledVarianceUniform(init_ops.Initializer):
  """Initializer that generates tensors with a Uniform distribution scaled as per https://github.com/torch/nn/blob/master/Linear.lua

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
    self.factor = factor
    self.seed = seed
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype

    if shape:
      n = float(shape[-1])
    else:
      n = 1.0

    self.stddev = np.sqrt(self.factor * 3.0 / n)
    return random_ops.random_uniform(shape, minval=-self.stddev, maxval=self.stddev, dtype=dtype, seed=self.seed)

  def get_config(self):
    return {"mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
            "dtype": self.dtype.name}

class ScaledVarianceRandomNormal(init_ops.Initializer):
  """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.factor = factor
    self.seed = seed
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype

    if shape:
      n = float(shape[-1])
    else:
      n = 1.0
    for dim in shape[:-2]:
      n *= float(dim)

    self.stddev = np.sqrt(self.factor * 2.0 / n)
    return random_ops.random_normal(shape, self.mean, self.stddev,
                                    dtype, seed=self.seed)

  def get_config(self):
    return {"mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
            "dtype": self.dtype.name}

class ConvLayer(object):

    def __init__(self, scope, num_outputs, kernel_size, padding='SAME', dropout=False, stride=1, normalizer_fn=False,
                 activation_fn=False, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=None):

        self.scope = scope
        self.dropout = dropout
        self.padding = padding
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalizer_fn = normalizer_fn
        self.activation_fn=activation_fn
        self.weights_initializer = weights_initializer
        self.weights_regularizer = weights_regularizer

    def apply(self, h):
        if self.activation_fn == False:
            if self.normalizer_fn==False:
                if self.dropout==False:
                    h_out = slim.conv2d(h, num_outputs=self.num_outputs, kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding, weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
                else:
                    h_out = slim.conv2d(slim.dropout(h, scope=self.scope+'-dropout'), num_outputs=self.num_outputs, kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding, weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
            else:
                if self.dropout == False:
                    h_out = slim.conv2d(h, num_outputs=self.num_outputs, kernel_size=self.kernel_size,
                                        stride=self.stride, scope=self.scope, padding=self.padding,
                                        normalizer_fn=self.normalizer_fn, weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
                else:
                    h_out = slim.conv2d(slim.dropout(h, scope=self.scope + '-dropout'), num_outputs=self.num_outputs,
                                        kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding, normalizer_fn=self.normalizer_fn, weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
        else:
            if self.normalizer_fn==False:
                if self.dropout==False:
                    h_out = slim.conv2d(h, num_outputs=self.num_outputs, kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding, activation_fn=self.activation_fn,
                                        weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
                else:
                    h_out = slim.conv2d(slim.dropout(h, scope=self.scope+'-dropout'), num_outputs=self.num_outputs, kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding,
                                        activation_fn=self.activation_fn,
                                        weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
            else:
                if self.dropout == False:
                    h_out = slim.conv2d(h, num_outputs=self.num_outputs, kernel_size=self.kernel_size,
                                        stride=self.stride, scope=self.scope, padding=self.padding,
                                        normalizer_fn=self.normalizer_fn, activation_fn=self.activation_fn,
                                        weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
                else:
                    h_out = slim.conv2d(slim.dropout(h, scope=self.scope + '-dropout'), num_outputs=self.num_outputs,
                                        kernel_size=self.kernel_size, stride=self.stride, scope=self.scope,
                                        padding=self.padding, normalizer_fn=self.normalizer_fn, activation_fn=self.activation_fn,
                                        weights_initializer=self.weights_initializer, weights_regularizer = self.weights_regularizer)
        return h_out


class ProjectionAdaptor(object):
    def __init__(self, scope, projection_width, num_outputs, dropout=False):
        self.dim_reduction_layer = ConvLayer(num_outputs=projection_width, kernel_size=1, stride=1, padding='SAME',
                                             weights_initializer=ScaledVarianceRandomNormal(),
                                             weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY),
                                             dropout=dropout, scope=scope + '/adapter/dim_reduction')
        self.output_layer = ConvLayer(num_outputs=num_outputs, kernel_size=1, stride=1, padding='SAME',
                                      weights_initializer=ScaledVarianceRandomNormal(),
                                      weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY),
                                      dropout=dropout, scope=scope + '/adapter/output', normalizer_fn=None, activation_fn=None)

    def apply(self, h):
        reduced_space = self.dim_reduction_layer.apply(h)
        return self.output_layer.apply(reduced_space)


def split(input_layer, stride, bottleneck_depth):
    '''
    The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
    64] filter, and then conv the result by [3, 3, 64]. Return the
    final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]

    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
    '''

    input_depth = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope('bneck_%d_1x1_%dd' %(input_depth, bottleneck_depth)):
        bneck_1x1 = slim.conv2d(input_layer, num_outputs=bottleneck_depth, kernel_size=1, stride=1,
                            padding='SAME',
                            weights_initializer=ScaledVarianceRandomNormal(),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))
    with tf.variable_scope('bneck_%d_3x3_%dd' %(bottleneck_depth, bottleneck_depth)):
        bneck_3x3 = slim.conv2d(bneck_1x1, num_outputs=bottleneck_depth, kernel_size=3, stride=stride,
                                padding='SAME',
                                weights_initializer=ScaledVarianceRandomNormal(),
                                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))

    return bneck_3x3

def bottleneck_b(input_layer, stride, cardinality, bottleneck_depth):
    '''
    The bottleneck strucutre in Figure 3b. Concatenates all the splits
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    nInputPlane = input_layer.get_shape().as_list()[-1]

    split_list = []
    for i in range(cardinality):
        with tf.variable_scope('split_%i'%i):
            splits = split(input_layer=input_layer, stride=stride, bottleneck_depth=bottleneck_depth)
        split_list.append(splits)

    # Concatenate splits and check the dimension
    concat_bottleneck = tf.concat(values=split_list, axis=3, name='concat_splits')

    return concat_bottleneck


class ResNextAdaptor(object):
    '''
        The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
        the tensor and restores the depth. Finally adds the identity and ReLu.
        '''

    def __init__(self, scope, cardinality, output_depth, num_filters, stride, dropout=False):
        self.scope = scope
        self.dropout = dropout
        self.num_filters = num_filters
        self.output_depth = output_depth
        self.cardinality = cardinality
        self.stride = stride

    def apply(self, input_layer):
        input_depth = input_layer.get_shape().as_list()[-1]

        with tf.variable_scope(self.scope):

            bottleneck_out = bottleneck_b(input_layer, stride=self.stride, bottleneck_depth=self.num_filters,
                                          cardinality=self.cardinality)

            restored = slim.conv2d(bottleneck_out, num_outputs=self.output_depth, kernel_size=1, stride=1,
                                   scope='restore_num_outputs', padding='SAME', activation_fn=None,
                                   weights_initializer=ScaledVarianceRandomNormal(),
                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))

            with tf.variable_scope('shortcut'):
                if input_depth != self.output_depth:
                    padded_input = slim.conv2d(input_layer, num_outputs=self.output_depth, kernel_size=1, stride=self.stride,
                                               padding='SAME', activation_fn=None,
                                               weights_initializer=ScaledVarianceRandomNormal(),
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))
                else:
                    padded_input = input_layer

            residual = tf.add(restored, padded_input, name='residual')
        return residual

class ResNextBlock(object):
    '''
    The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
    the tensor and restores the depth. Finally adds the identity and ReLu.
    '''
    def __init__(self, scope, cardinality, bottleneck_depth, num_filters, stride, dropout=False):
        self.scope = scope
        self.dropout = dropout
        self.num_filters = num_filters
        self.bottleneck_depth = bottleneck_depth
        self.cardinality = cardinality
        self.stride = stride

    def apply(self, input_layer):
        input_depth = input_layer.get_shape().as_list()[-1]
        # output width 4*self.num_filters as per line 96 in
        # https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua, commit 833a384
        output_depth = 4*self.num_filters

        with tf.variable_scope(self.scope):

            bottleneck_out = bottleneck_b(input_layer, stride=self.stride, bottleneck_depth = self.num_filters, cardinality=self.cardinality)


            restored = slim.conv2d(bottleneck_out, num_outputs=output_depth, kernel_size=1, stride=1,
                                   scope='restore_num_outputs', padding='SAME', activation_fn=None,
                                   weights_initializer=ScaledVarianceRandomNormal(),
                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))

            with tf.variable_scope('shortcut'):
                if input_depth != output_depth:
                    padded_input = slim.conv2d(input_layer, num_outputs=output_depth, kernel_size=1, stride=self.stride,
                                               padding='SAME', activation_fn=None,
                                               weights_initializer=ScaledVarianceRandomNormal(),
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))
                else:
                    padded_input = input_layer

            residual = tf.add(restored, padded_input, name='residual')
            # output = tf.nn.relu(residual, name='residual_relu')
            output = residual

        return output

class ResNextGroup(object):

    def __init__(self, scope, num_blocks, num_filters, bottleneck_depth, cardinality, stride, dropout=False):
        self.scope = scope
        self.dropout = dropout
        self.num_filters = num_filters
        self.cardinality = cardinality
        self.num_blocks = num_blocks
        self.bottleneck_depth = bottleneck_depth
        self.stride = stride

    def apply(self, h):
        tensor_stack = [h]
        with tf.variable_scope(self.scope):
            for i in range(self.num_blocks):
                if i == 0:
                    stride=self.stride
                else:
                    stride=1
                h = ResNextBlock(num_filters=self.num_filters, cardinality=self.cardinality, bottleneck_depth = self.bottleneck_depth,
                                 stride=stride, dropout=self.dropout, scope='block%d' % i).apply(tensor_stack[-1])
                tensor_stack.append(h)
            return tensor_stack[-1]

class AveragePoolLayer(object):
    def __init__(self, scope, axis, keep_dims):
        self.scope = scope
        self.axis=axis
        self.keep_dims=keep_dims

    def apply(self, h):
        with tf.variable_scope(self.scope):
            average_pool = tf.reduce_mean(h, axis=self.axis, keep_dims=self.keep_dims)
            return average_pool


# The ResNext architecture is based on the following code:
# https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/resNeXt.py
# commit 8a00577495fb01cf98bf77562422390b652e1a4e
# ResNeXt. total layers = 1 + 3n + 3n + 3n +1 = 9n + 2

ArchitectureResNext = [
    ConvLayer(num_outputs=64, kernel_size=3, stride=1, scope='conv0',
              activation_fn=None,
              weights_initializer=ScaledVarianceRandomNormal(),
              weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)),

    ResNextBlock(num_filters=64, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group1/block1'),
    ResNextBlock(num_filters=64, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group1/block2'),
    ResNextBlock(num_filters=64, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group1/block3'),

    ResNextBlock(num_filters=128, cardinality=2, bottleneck_depth=64, stride=2, dropout=False, scope='bottleneck_group2/block1'),
    ResNextBlock(num_filters=128, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group2/block2'),
    ResNextBlock(num_filters=128, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group2/block3'),

    ResNextBlock(num_filters=256, cardinality=2, bottleneck_depth=64, stride=2, dropout=False, scope='bottleneck_group3/block1'),
    ResNextBlock(num_filters=256, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group3/block2'),
    ResNextBlock(num_filters=256, cardinality=2, bottleneck_depth=64, stride=1, dropout=False, scope='bottleneck_group3/block3'),

    AveragePoolLayer(scope='avg_pool', axis=[1,2], keep_dims=True),
    ConvLayer(num_outputs=10, kernel_size=1, stride=1, normalizer_fn=None, activation_fn=None,
              weights_initializer=ScaledVarianceUniform(),
              weights_regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY), scope='logits/fc'),
]

AdaptorStack = [
    None,

    ResNextAdaptor(cardinality=2, output_depth=256, num_filters=64/4, stride=1, dropout=False, scope='bottleneck_group1/adaptor1'),
    ResNextAdaptor(cardinality=2, output_depth=256, num_filters=64/4, stride=1, dropout=False, scope='bottleneck_group1/adaptor2'),
    ResNextAdaptor(cardinality=2, output_depth=256, num_filters=64/4, stride=1, dropout=False, scope='bottleneck_group1/adaptor3'),

    ResNextAdaptor(cardinality=2, output_depth=512, num_filters=128/4, stride=2, dropout=False, scope='bottleneck_group2/adaptor1'),
    ResNextAdaptor(cardinality=2, output_depth=512, num_filters=128/4, stride=1, dropout=False, scope='bottleneck_group2/adaptor2'),
    ResNextAdaptor(cardinality=2, output_depth=512, num_filters=128/4, stride=1, dropout=False, scope='bottleneck_group2/adaptor3'),

    ResNextAdaptor(cardinality=2, output_depth=1024, num_filters=256/4, stride=2, dropout=False, scope='bottleneck_group3/adaptor1'),
    ResNextAdaptor(cardinality=2, output_depth=1024, num_filters=256/4, stride=1, dropout=False, scope='bottleneck_group3/adaptor2'),
    ResNextAdaptor(cardinality=2, output_depth=1024, num_filters=256/4, stride=1, dropout=False, scope='bottleneck_group3/adaptor3'),

    None,
    ProjectionAdaptor(projection_width=1024/8, num_outputs=10, dropout=False, scope='logits/adaptor')
]