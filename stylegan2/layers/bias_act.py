import tensorflow as tf


from stylegan2.layers.cuda.fused_bias_act import fused_bias_act


class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        self.lrmul = lrmul
        self.act = act

    def build(self, input_shape):
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b
        x = fused_bias_act(inputs, b=b, act=self.act, alpha=None, gain=None)
        return x

    def get_config(self):
        config = super(BiasAct, self).get_config()
        config.update({
            'lrmul': self.lrmul,
            'act': self.act,
        })
        return config
