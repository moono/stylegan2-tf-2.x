import tensorflow as tf

from stylegan2.layers.commons import compute_runtime_coef


class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain, lrmul, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        fan_in = tf.reduce_prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w

        c = tf.reduce_prod(tf.shape(inputs)[1:])
        x = tf.reshape(inputs, shape=[-1, c])
        x = tf.matmul(x, weight)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'runtime_coef': self.runtime_coef,
        })
        return config
