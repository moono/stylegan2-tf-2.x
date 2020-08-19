import numpy as np
import tensorflow as tf


# the custom cuda compiled fused_bias_act is not compatible with tf.GradientTape()
# I think it's because of 2nd order gradient computation
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


# import tensorflow as tf
#
#
# from stylegan2.layers.cuda.fused_bias_act import fused_bias_act
#
#
# class BiasAct(tf.keras.layers.Layer):
#     def __init__(self, lrmul, act, **kwargs):
#         super(BiasAct, self).__init__(**kwargs)
#         self.lrmul = lrmul
#         self.act = act
#
#     def build(self, input_shape):
#         b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
#         self.b = tf.Variable(b_init, name='b', trainable=True)
#
#     def call(self, inputs, training=None, mask=None):
#         b = self.lrmul * self.b
#         x = fused_bias_act(inputs, b=b, act=self.act, alpha=None, gain=None)
#         return x
#
#     def get_config(self):
#         config = super(BiasAct, self).get_config()
#         config.update({
#             'lrmul': self.lrmul,
#             'act': self.act,
#         })
#         return config
