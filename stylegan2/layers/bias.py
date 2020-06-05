import tensorflow as tf


class Bias(tf.keras.layers.Layer):
    def __init__(self, lrmul, n_dims, **kwargs):
        super(Bias, self).__init__(**kwargs)
        assert n_dims == 2 or n_dims == 4

        self.lrmul = lrmul
        self.n_dims = n_dims

        if self.n_dims == 2:
            self.reshape = tf.keras.layers.Lambda(lambda x: x)
        else:
            self.reshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [1, -1, 1, 1]))

    def build(self, input_shape):
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b
        b = self.reshape(b)
        x = inputs + b
        return x

    def get_config(self):
        config = super(Bias, self).get_config()
        config.update({
            'lrmul': self.lrmul,
            'n_dims': self.n_dims,
        })
        return config
