import tensorflow as tf

from stylegan2.layers.modulated_conv2d import ModulatedConv2D
from stylegan2.layers.bias_act import BiasAct
from stylegan2.layers.noise import Noise
from stylegan2.layers.to_rgb import ToRGB
from stylegan2.layers.cuda.upfirdn_2d_v2 import upsample_2d, compute_paddings


class SynthesisConstBlock(tf.keras.layers.Layer):
    def __init__(self, fmaps, res, **kwargs):
        super(SynthesisConstBlock, self).__init__(**kwargs)
        assert res == 4
        self.res = res
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        # conv block
        self.conv = ModulatedConv2D(in_res=res, in_fmaps=self.fmaps, fmaps=self.fmaps, kernel=3, up=False, down=False,
                                    demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                    fused_modconv=True, name='conv')
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias')

    def build(self, input_shape):
        # starting const variable
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        const_init = tf.random.normal(shape=(1, self.fmaps, self.res, self.res), mean=0.0, stddev=1.0)
        self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, training=None, mask=None):
        w0 = inputs
        batch_size = tf.shape(w0)[0]

        # const block
        x = tf.tile(self.const, [batch_size, 1, 1, 1])

        # conv block
        x = self.conv([x, w0])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)
        return x

    def get_config(self):
        config = super(SynthesisConstBlock, self).get_config()
        config.update({
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class SynthesisBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, fmaps, res, **kwargs):
        super(SynthesisBlock, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.fmaps = fmaps
        self.res = res
        self.gain = 1.0
        self.lrmul = 1.0

        # conv0 up
        self.conv_0 = ModulatedConv2D(in_res=res//2, in_fmaps=self.in_ch, fmaps=self.fmaps, kernel=3, up=True, down=False,
                                      demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                      fused_modconv=True, name='conv_0')
        self.apply_noise_0 = Noise(name='noise_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # conv block
        self.conv_1 = ModulatedConv2D(in_res=res, in_fmaps=self.fmaps, fmaps=self.fmaps, kernel=3, up=False, down=False,
                                      demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                      fused_modconv=True, name='conv_1')
        self.apply_noise_1 = Noise(name='noise_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

    def call(self, inputs, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_act_0(x)

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_act_1(x)
        return x

    def get_config(self):
        config = super(SynthesisBlock, self).get_config()
        config.update({
            'in_ch': self.in_ch,
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class Synthesis(tf.keras.layers.Layer):
    def __init__(self, resolutions, featuremaps, name, **kwargs):
        super(Synthesis, self).__init__(name=name, **kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        # self.resample_kernel = [1, 3, 3, 1]

        self.k, self.pad0, self.pad1 = compute_paddings([1, 3, 3, 1], None, up=True, down=False, is_conv=False)

        # initial layer
        res, n_f = resolutions[0], featuremaps[0]
        self.initial_block = SynthesisConstBlock(fmaps=n_f, res=res, name='{:d}x{:d}/const'.format(res, res))
        self.initial_torgb = ToRGB(in_ch=n_f, res=res, name='{:d}x{:d}/ToRGB'.format(res, res))

        # stack generator block with lerp block
        prev_n_f = n_f
        self.blocks = list()
        self.torgbs = list()
        for res, n_f in zip(self.resolutions[1:], self.featuremaps[1:]):
            self.blocks.append(SynthesisBlock(in_ch=prev_n_f, fmaps=n_f, res=res,
                                              name='{:d}x{:d}/block'.format(res, res)))
            self.torgbs.append(ToRGB(in_ch=n_f, res=res * 2, name='{:d}x{:d}/ToRGB'.format(res, res)))
            prev_n_f = n_f

    def call(self, inputs, training=None, mask=None):
        w_broadcasted = inputs

        # initial layer
        w0, w1 = w_broadcasted[:, 0], w_broadcasted[:, 1]
        x = self.initial_block(w0)
        y = self.initial_torgb([x, w1])

        layer_index = 1
        for block, torgb in zip(self.blocks, self.torgbs):
            w0 = w_broadcasted[:, layer_index]
            w1 = w_broadcasted[:, layer_index + 1]
            w2 = w_broadcasted[:, layer_index + 2]

            y_res = block.res // 2
            x = block([x, w0, w1])
            y = upsample_2d(y, y_res, self.pad0, self.pad1, self.k)
            y = y + torgb([x, w2])

            layer_index += 2

        images_out = y
        return images_out

    def get_config(self):
        config = super(Synthesis, self).get_config()
        config.update({
            'resolutions': self.resolutions,
            'featuremaps': self.featuremaps,
            'k': self.k,
            'pad0': self.pad0,
            'pad1': self.pad1,
        })
        return config
