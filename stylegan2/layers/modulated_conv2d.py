import tensorflow as tf

from stylegan2.layers.commons import compute_runtime_coef
from stylegan2.layers.dense import Dense
from stylegan2.layers.bias_act import BiasAct
from stylegan2.layers.cuda.upfirdn_2d import upsample_conv_2d, conv_downsample_2d


class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, in_fmaps, fmaps, kernel, up, down, demodulate, resample_kernel, gain, lrmul, fused_modconv, **kwargs):
        super(ModulatedConv2D, self).__init__(**kwargs)
        assert not (up and down)
        self.in_fmaps = in_fmaps
        self.fmaps = fmaps
        self.kernel = kernel
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.fused_modconv = fused_modconv
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.lrmul = lrmul

        # self.factor = 2
        self.mod_dense = Dense(self.in_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias_act = BiasAct(lrmul=1.0, n_dims=2, act='linear', name='mod_bias')

    def build(self, input_shape):
        # x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = [self.kernel, self.kernel, self.in_fmaps, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def modulate(self, y):
        # [BkkIO] Introduce minibatch dimension
        w = self.runtime_coef * self.w
        ww = w[tf.newaxis]

        # Modulate
        s = self.mod_dense(y)           # [BI]
        s = self.mod_bias_act(s) + 1.0  # [BI]
        ww *= tf.cast(s[:, tf.newaxis, tf.newaxis, :, tf.newaxis], w.dtype)  # [BkkIO]
        return ww

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        # height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare weights: [BkkIO] Introduce minibatch dimension
        w = self.runtime_coef * self.w
        ww = w[tf.newaxis]

        # Modulate
        s = self.mod_dense(y)           # [BI]
        s = self.mod_bias_act(s) + 1.0  # [BI]
        ww *= s[:, tf.newaxis, tf.newaxis, :, tf.newaxis]  # [BkkIO]

        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + 1e-8)  # [BO]
            ww *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, :]                       # [BkkIO]

        if self.fused_modconv:
            # Fused => reshape minibatch to convolution groups
            x_shape = tf.shape(x)
            ww_shape = tf.shape(ww)
            x = tf.reshape(x, [1, -1, x_shape[2], x_shape[3]])
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww_shape[1], ww_shape[2], ww_shape[3], -1])
        else:
            # [BIhw] Not fused => scale input activations
            x *= s[:, :, tf.newaxis, tf.newaxis]

        # Convolution with optional up/downsampling.
        if self.up:
            x = upsample_conv_2d(x, w, data_format='NCHW', k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, w, data_format='NCHW', k=self.resample_kernel)
        else:
            x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # Reshape/scale output
        if self.fused_modconv:
            # Fused => reshape convolution groups back to minibatch
            x_shape = tf.shape(x)
            x = tf.reshape(x, [-1, self.fmaps, x_shape[2], x_shape[3]])
        elif self.demodulate:
            # [BOhw] Not fused => scale output activations
            x *= d[:, :, tf.newaxis, tf.newaxis]
        return x