import numpy as np
import tensorflow as tf


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, gain=np.sqrt(2), **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2
        self.gain = gain

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
