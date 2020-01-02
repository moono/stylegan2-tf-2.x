import numpy as np
import tensorflow as tf

from stylegan2.image_proc import blur2d


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


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, is_down, fmaps, kernel, gain, lrmul, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.is_down = is_down
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul

        if self.is_down:
            self.downscale2d = Downscale2D()
        else:
            self.downscale2d = tf.keras.layers.Lambda(lambda x: tf.identity(x))

    def build(self, input_shape):
        assert len(input_shape) == 4
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(initial_value=w_init(shape=weight_shape, dtype='float32'), trainable=True, name='w')

    def call(self, inputs, training=None, mask=None):
        w = self.runtime_coef * self.w

        x = self.downscale2d(inputs)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
        return x


class Bias(tf.keras.layers.Layer):
    def __init__(self, lrmul, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.lrmul = lrmul

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4

        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b

        if len(tf.shape(inputs)) == 2:
            x = inputs + b
        elif len(tf.shape(inputs)) == 4:
            x = inputs + tf.reshape(b, [1, -1, 1, 1])
        else:
            raise ValueError('Wrong dimension!!')
        return x


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.gain = np.sqrt(2)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        x *= self.gain
        return x


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


class Upscale2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Upscale2D, self).__init__(**kwargs)

        self.resize = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='bilinear')
        self.blur = tf.keras.layers.Lambda(lambda x: blur2d(x, [1, 2, 1]))

    def call(self, inputs, training=None, mask=None):
        x = self.resize(inputs)
        x = self.blur(x)
        return x


class Downscale2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Downscale2D, self).__init__(**kwargs)

        self.blur = tf.keras.layers.Lambda(lambda x: blur2d(x, [1, 2, 1]))
        self.resize = tf.keras.layers.AveragePooling2D((2, 2), strides=None, padding='same',
                                                       data_format='channels_first')

    def call(self, inputs, training=None, mask=None):
        x = self.blur(inputs)
        x = self.resize(x)
        return x


class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, is_up, do_demod, in_channel, fmaps, kernel, gain, lrmul, **kwargs):
        super(ModulatedConv2D, self).__init__(**kwargs)
        self.is_up = is_up
        self.do_demod = do_demod
        self.in_channel = in_channel
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul

        self.mod_dense = Dense(self.in_channel, gain, lrmul, name='mod_dense')
        self.mod_bias = Bias(lrmul, name='mod_bias')
        self.demod_scale = tf.keras.layers.Lambda(lambda x: tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3]) + 1e-8))
        if self.is_up:
            self.upscale2d = Upscale2D()
        else:
            self.upscale2d = tf.keras.layers.Lambda(lambda x: tf.identity(x))

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = [self.kernel, self.kernel, x_shape[1], self.fmaps]
        init_std, runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def modulation(self, w, weight):
        # transform incoming W to style
        style = self.mod_dense(w)
        style = self.mod_bias(style) + 1.0

        # scale weight
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]
        return weight

    def demodulation(self, weight):
        d = self.demod_scale(weight)
        weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]
        return weight

    def prepare_weights(self, w):
        # prepare convolution kernel weights: (3, 3, in_channel, fmaps) -> (1, 3, 3, in_channel, fmaps)
        weight = self.runtime_coef * self.w
        weight = weight[np.newaxis]

        # modulation: (?, 3, 3, in_channel, fmaps)
        weight = self.modulation(w, weight)

        # demodulation: : (?, 3, 3, in_channel, fmaps)
        if self.do_demod:
            weight = self.demodulation(weight)

        # weight: reshape, prepare for fused operation
        # (?, 3, 3, in_channel, fmaps) -> (3, 3, in_channel, ?, fmaps) -> (3, 3, in_channel, ? * fmaps)
        weight_new_shape = [weight.shape[1], weight.shape[2], weight.shape[3], -1]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])
        weight = tf.reshape(weight, shape=weight_new_shape)
        return weight

    def call(self, inputs, training=None, mask=None):
        x, w = inputs

        # prepare convolution kernel weights
        weight = self.prepare_weights(w)

        # prepare inputs: reshape minibatch to convolution groups
        # (?, in_channel, h, w) -> (1, ? * in_channel, h, w)
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])

        # upsample & blur if needed
        x = self.upscale2d(x)

        # actual conv
        x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # x: reshape back
        x = tf.reshape(x, [-1, self.fmaps, x.shape[2], x.shape[3]])
        return x


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
