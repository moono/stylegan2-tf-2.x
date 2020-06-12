import tensorflow as tf

from stylegan2.layers.conv import Conv2D
from stylegan2.layers.bias_act import BiasAct


class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, res, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.res = res

        self.conv = Conv2D(in_res=res, in_fmaps=3, fmaps=self.fmaps, kernel=1, up=False, down=False,
                           resample_kernel=None, gain=1.0, lrmul=1.0, name='conv')
        self.apply_bias_act = BiasAct(lrmul=1.0, act='lrelu', name='bias')

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias_act(y)
        return y

    def get_config(self):
        config = super(FromRGB, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'res': self.res,
        })
        return config
