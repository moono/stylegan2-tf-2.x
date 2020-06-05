import numpy as np
import tensorflow as tf

from stylegan2.layers.commons import compute_runtime_coef
from stylegan2.layers.dense import Dense
from stylegan2.layers.bias import Bias
from stylegan2.layers.upfirdn_2d import setup_resample_kernel, upsample_conv_2d, conv_downsample_2d


class FusedModConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, style_fmaps, demodulate, up, down, resample_kernel, **kwargs):
        super(FusedModConv, self).__init__(**kwargs)
        assert not (up and down)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.style_fmaps = style_fmaps
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = setup_resample_kernel(k=resample_kernel)

        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = Bias(lrmul=1.0, n_dims=2, name='mod_bias')

        # set proper conv type
        if self.up:
            self.conv = tf.keras.layers.Lambda(lambda x: upsample_conv_2d(x[0], self.k, x[1], self.factor, self.gain))
        elif self.down:
            self.conv = tf.keras.layers.Lambda(lambda x: conv_downsample_2d(x[0], self.k, x[1], self.factor, self.gain))
        else:
            self.conv = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(x[0], x[1], data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME'))

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = [self.kernel, self.kernel, x_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        # convolution kernel weights for fused conv
        weight = self.runtime_coef * self.w     # [kkIO]
        weight = weight[np.newaxis]             # [BkkIO]

        # modulation
        style = self.mod_dense(w)                                   # [BI]
        style = self.mod_bias(style) + 1.0                          # [BI]
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]   # [BkkIO]

        # demodulation
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]                       # [BkkIO]

        # weight: reshape, prepare for fused operation
        new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]      # [kkI(BO)]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])                                              # [kkIBO]
        weight = tf.reshape(weight, shape=new_weight_shape)                                         # [kkI(BO)]
        return weight

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare convolution kernel weights
        weight = self.scale_conv_weights(w)

        # prepare inputs: reshape minibatch to convolution groups
        x = tf.reshape(x, [1, -1, height, width])

        # actual conv
        x = self.conv([x, weight])

        # x: reshape back
        x = tf.reshape(x, [-1, self.fmaps, tf.shape(x)[2], tf.shape(x)[3]])
        return x

    def get_config(self):
        config = super(FusedModConv, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'style_fmaps': self.style_fmaps,
            'demodulate': self.demodulate,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config
