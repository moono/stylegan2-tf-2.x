import tensorflow as tf

from stylegan2.layers.commons import compute_runtime_coef
from stylegan2.layers.cuda.upfirdn_2d import upsample_conv_2d, conv_downsample_2d


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, up, down, resample_kernel, gain, lrmul, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel

    def build(self, input_shape):
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        weight = self.runtime_coef * self.w

        # actual conv
        if self.up:
            x = upsample_conv_2d(x, weight, data_format='NCHW', k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, weight, data_format='NCHW', k=self.resample_kernel)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return x

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'up': self.up,
            'down': self.down,
            'resample_kernel': self.resample_kernel,
        })
        return config
