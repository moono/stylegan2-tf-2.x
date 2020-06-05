import tensorflow as tf

from stylegan2.layers.resize_conv import ResizeConv2D
from stylegan2.layers.bias import Bias
from stylegan2.layers.leaky_relu import LeakyReLU


class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps

        self.conv = ResizeConv2D(fmaps=self.fmaps, kernel=1, gain=1.0, lrmul=1.0,
                                 up=False, down=False, resample_kernel=None, name='conv')
        self.apply_bias = Bias(lrmul=1.0, n_dims=4, name='bias')
        self.leaky_relu = LeakyReLU(name='lrelu')

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias(y)
        y = self.leaky_relu(y)
        return y
