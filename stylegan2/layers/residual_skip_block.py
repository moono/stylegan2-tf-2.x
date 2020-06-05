import numpy as np
import tensorflow as tf

from stylegan2.layers.dense import Dense
from stylegan2.layers.bias import Bias
from stylegan2.layers.leaky_relu import LeakyReLU
from stylegan2.layers.embedding import LabelEmbedding
from stylegan2.layers.resize_conv import ResizeConv2D
from stylegan2.layers.mini_batch_std import MinibatchStd


class ResidualSkipBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, res, **kwargs):
        super(ResidualSkipBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.res = res
        self.resnet_scale = 1. / np.sqrt(2.)

        # conv_0
        self.conv_0 = ResizeConv2D(fmaps=self.n_f0, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=False, resample_kernel=None, name='conv_0')
        self.apply_bias_0 = Bias(self.lrmul, n_dims=4, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # conv_1 down
        self.conv_1 = ResizeConv2D(fmaps=self.n_f1, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=True, resample_kernel=[1, 3, 3, 1], name='conv_1')
        self.apply_bias_1 = Bias(self.lrmul, n_dims=4, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

        # label embeddings
        self.label_embedding = LabelEmbedding(embed_dim=self.n_f1)
        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8))

        # resnet skip
        self.conv_skip = ResizeConv2D(fmaps=self.n_f1, kernel=1, gain=self.gain, lrmul=self.lrmul,
                                      up=False, down=True, resample_kernel=[1, 3, 3, 1], name='skip')

    def call(self, inputs, training=None, mask=None):
        x, labels = inputs
        residual = x

        # conv0
        x = self.conv_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # conv1 down
        x = self.conv_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)

        # label embedding
        h = tf.reduce_mean(x, axis=[2, 3])
        y = self.label_embedding(labels)
        y = self.normalize(y)
        p = tf.reduce_sum(y * h, axis=1, keepdims=True)

        # resnet skip
        residual = self.conv_skip(residual)
        x = (x + residual) * self.resnet_scale
        return x, p


class ResidualSkipLastBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, res, **kwargs):
        super(ResidualSkipLastBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.res = res

        self.minibatch_std = MinibatchStd(group_size=4, num_new_features=1, name='minibatchstd')

        # conv_0
        self.conv_0 = ResizeConv2D(fmaps=self.n_f0, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=False, resample_kernel=None, name='conv_0')
        self.apply_bias_0 = Bias(self.lrmul, n_dims=4, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # dense_1
        self.dense_1 = Dense(self.n_f1, gain=self.gain, lrmul=self.lrmul, name='dense_1')
        self.apply_bias_1 = Bias(self.lrmul, n_dims=2, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

    def call(self, x, training=None, mask=None):
        x = self.minibatch_std(x)

        # conv_0
        x = self.conv_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # dense_1
        x = self.dense_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)
        return x
