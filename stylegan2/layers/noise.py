import tensorflow as tf


class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.zeros(shape=(), dtype=tf.dtypes.float32)
        self.noise_strength = tf.Variable(w_init, trainable=True, name='w')

    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        # noise: [1, 1, x_shape[2], x_shape[3]] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.dtypes.float32)

        x = inputs + noise * self.noise_strength
        return x

    def get_config(self):
        config = super(Noise, self).get_config()
        config.update({})
        return config
