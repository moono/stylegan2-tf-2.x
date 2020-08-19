import numpy as np
import tensorflow as tf

from stylegan2_ref.upfirdn_2d import setup_resample_kernel, upsample_conv_2d, conv_downsample_2d


def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef


class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain, lrmul, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4
        fan_in = np.prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.matmul(x, weight)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        assert act in ['linear', 'lrelu']
        self.lrmul = lrmul

        if act == 'linear':
            self.act = tf.keras.layers.Lambda(lambda x: tf.identity(x))
            self.gain = 1.0
        else:
            self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.gain = np.sqrt(2)

    def build(self, input_shape):
        self.len2 = True if len(input_shape) == 2 else False
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b

        if self.len2:
            x = inputs + b
        else:
            x = inputs + tf.reshape(b, shape=[1, -1, 1, 1])
        x = self.act(x)
        x = self.gain * x
        return x

    def get_config(self):
        config = super(BiasAct, self).get_config()
        config.update({
            'lrmul': self.lrmul,
            'gain': self.gain,
            'len2': self.len2,
        })
        return config


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2
        self.gain = np.sqrt(2)

        self.act = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        x *= self.gain
        return x

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gain': self.gain,
        })
        return config


class LabelEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.w)
        return x

    def get_config(self):
        config = super(LabelEmbedding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config


class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.dtypes.float32, trainable=True, name='w')

    def call(self, x, training=None, mask=None):
        x_shape = tf.shape(x)
        noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.dtypes.float32)

        x += noise * self.noise_strength
        return x


class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x, training=None, mask=None):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = tf.shape(x)

        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        return tf.concat([x, y], axis=1)


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
        self.mod_bias = BiasAct(lrmul=1.0, act='linear', name='mod_bias')

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

        if self.up:
            x = upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

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

    def build(self, input_shape):
        assert len(input_shape) == 4
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(initial_value=w_init(shape=weight_shape, dtype='float32'), trainable=True, name='w')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        weight = self.runtime_coef * self.w

        if self.up:
            x = upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
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
