import tensorflow as tf

from stylegan2.layers.modulated_conv2d import ModulatedConv2D
from stylegan2.layers.bias_act import BiasAct


class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_ch, res, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.res = res

        self.conv = ModulatedConv2D(in_res=res, in_fmaps=in_ch, fmaps=3, kernel=1, up=False, down=False, demodulate=False,
                                    resample_kernel=None, gain=1.0, lrmul=1.0, fused_modconv=True, name='conv')
        self.apply_bias = BiasAct(lrmul=1.0, act='linear', name='bias')

    def call(self, inputs, training=None, mask=None):
        x, w = inputs

        x = self.conv([x, w])
        x = self.apply_bias(x)
        return x

    def get_config(self):
        config = super(ToRGB, self).get_config()
        config.update({
            'in_ch': self.in_ch,
            'res': self.res,
        })
        return config
