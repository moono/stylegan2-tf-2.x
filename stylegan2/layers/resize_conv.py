import tensorflow as tf

from stylegan2.layers.commons import compute_runtime_coef
from stylegan2.layers.upfirdn_2d import setup_resample_kernel, upsample_conv_2d, conv_downsample_2d


class ResizeConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, up, down, resample_kernel, **kwargs):
        super(ResizeConv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = setup_resample_kernel(k=resample_kernel)

        # set proper conv type
        if self.up:
            self.conv = tf.keras.layers.Lambda(lambda x: upsample_conv_2d(x[0], self.k, x[1], self.factor, self.gain))
        elif self.down:
            self.conv = tf.keras.layers.Lambda(lambda x: conv_downsample_2d(x[0], self.k, x[1], self.factor, self.gain))
        else:
            self.conv = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(x[0], x[1], data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME'))

    def build(self, input_shape):
        assert len(input_shape) == 4
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        weight = self.runtime_coef * self.w

        # actual conv
        x = self.conv([x, weight])
        return x

    def get_config(self):
        config = super(ResizeConv2D, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config
