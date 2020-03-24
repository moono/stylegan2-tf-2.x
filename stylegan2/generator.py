import numpy as np
import tensorflow as tf

from stylegan2.utils import lerp
from stylegan2.upfirdn_2d import setup_resample_kernel, upsample_2d
from stylegan2.custom_layers import LabelEmbedding, Dense, Bias, LeakyReLU, Noise, FusedModConv


class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_ch, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.conv = FusedModConv(fmaps=3, kernel=1, gain=1.0, lrmul=1.0, style_fmaps=self.in_ch,
                                 demodulate=False, up=False, down=False, resample_kernel=None, name='conv')
        self.apply_bias = Bias(lrmul=1.0, name='bias')

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        assert x.shape[1] == self.in_ch

        x = self.conv([x, w])
        x = self.apply_bias(x)
        return x

    def get_config(self):
        config = super(ToRGB, self).get_config()
        config.update({
            'in_ch': self.in_ch,
        })
        return config


class Mapping(tf.keras.layers.Layer):
    def __init__(self, w_dim, labels_dim, n_mapping, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.labels_dim = labels_dim
        self.n_mapping = n_mapping
        self.gain = 1.0
        self.lrmul = 0.01

        if self.labels_dim > 0:
            self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim, name='labels_embedding')

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8))

        self.dense_layers = list()
        self.bias_layers = list()
        self.act_layers = list()
        for ii in range(self.n_mapping):
            self.dense_layers.append(Dense(w_dim, gain=self.gain, lrmul=self.lrmul, name='dense_{:d}'.format(ii)))
            self.bias_layers.append(Bias(lrmul=self.lrmul, name='bias_{:d}'.format(ii)))
            self.act_layers.append(LeakyReLU(name='lrelu_{:d}'.format(ii)))

    def call(self, inputs, training=None, mask=None):
        latents, labels = inputs
        x = latents

        # embed label if any
        if self.labels_dim > 0:
            y = self.labels_embedding(labels)
            x = tf.concat([x, y], axis=1)

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias, leaky_relu in zip(self.dense_layers, self.bias_layers, self.act_layers):
            x = dense(x)
            x = apply_bias(x)
            x = leaky_relu(x)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'labels_dim': self.labels_dim,
            'n_mapping': self.n_mapping,
            'n_broadcast': self.n_broadcast,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config


class SynthesisConstBlock(tf.keras.layers.Layer):
    def __init__(self, fmaps, res, **kwargs):
        super(SynthesisConstBlock, self).__init__(**kwargs)
        assert res == 4
        self.res = res
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        # conv block
        self.conv = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                 demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv')
        self.apply_noise = Noise(name='noise')
        self.apply_bias = Bias(lrmul=self.lrmul, name='bias')
        self.leaky_relu = LeakyReLU(name='lrelu')

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
        x = self.apply_bias(x)
        x = self.leaky_relu(x)
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
        self.conv_0 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.in_ch,
                                   demodulate=True, up=True, down=False, resample_kernel=[1, 3, 3, 1], name='conv_0')
        self.apply_noise_0 = Noise(name='noise_0')
        self.apply_bias_0 = Bias(lrmul=self.lrmul, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # conv block
        self.conv_1 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                   demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv_1')
        self.apply_noise_1 = Noise(name='noise_1')
        self.apply_bias_1 = Bias(lrmul=self.lrmul, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

    def call(self, inputs, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)
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
        self.k = setup_resample_kernel(k=[1, 3, 3, 1])

        # initial layer
        res, n_f = resolutions[0], featuremaps[0]
        self.initial_block = SynthesisConstBlock(fmaps=n_f, res=res, name='{:d}x{:d}/const'.format(res, res))
        self.initial_torgb = ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res))

        # stack generator block with lerp block
        prev_n_f = n_f
        self.blocks = list()
        self.torgbs = list()
        for res, n_f in zip(self.resolutions[1:], self.featuremaps[1:]):
            self.blocks.append(SynthesisBlock(in_ch=prev_n_f, fmaps=n_f, res=res,
                                              name='{:d}x{:d}/block'.format(res, res)))
            self.torgbs.append(ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res)))
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

            x = block([x, w0, w1])
            y = upsample_2d(y, self.k, factor=2, gain=1.0)
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
        })
        return config


class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.z_dim = g_params['z_dim']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.n_mapping = g_params['n_mapping']
        self.resolutions = g_params['resolutions']
        self.featuremaps = g_params['featuremaps']
        self.w_ema_decay = g_params['w_ema_decay']
        self.style_mixing_prob = g_params['style_mixing_prob']

        self.n_broadcast = len(self.resolutions) * 2
        self.mixing_layer_indices = np.arange(self.n_broadcast)[np.newaxis, :, np.newaxis]

        self.g_mapping = Mapping(self.w_dim, self.labels_dim, self.n_mapping, name='g_mapping')
        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, np.newaxis], [1, self.n_broadcast, 1]))
        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')

    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(tf.zeros(shape=[self.w_dim], dtype=tf.dtypes.float32), name='w_avg', trainable=False)

    def set_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        def split_first_name(name):
            splitted = name.split('/')
            new_name = '/'.join(splitted[1:])
            return new_name

        for cw in self.trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(lerp(sw, cw, beta))
                    break

        for cw in self.non_trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.non_trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(lerp(sw, cw, beta_nontrainable))
                    break
        return

    def update_moving_average_of_w(self, w_broadcasted):
        # compute average of current w
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)

        # compute moving average of w and update(assign) w_avg
        self.w_avg.assign(lerp(batch_avg, self.w_avg, self.w_ema_decay))
        return

    def style_mixing_regularization(self, latents1, labels, w_broadcasted1):
        # get another w and broadcast it
        latents2 = tf.random.normal(shape=tf.shape(latents1), dtype=tf.dtypes.float32)
        dlatents2 = self.g_mapping([latents2, labels])
        w_broadcasted2 = self.broadcast(dlatents2)

        # find mixing limit index
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform([], 1, self.n_broadcast, dtype=tf.dtypes.int32)
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype=tf.dtypes.int32)

        # mix it
        mixed_w_broadcasted = tf.where(
            condition=tf.broadcast_to(self.mixing_layer_indices < mixing_cutoff_index, tf.shape(w_broadcasted1)),
            x=w_broadcasted1,
            y=w_broadcasted2)
        return mixed_w_broadcasted

    def truncation_trick(self, w_broadcasted, truncation_cutoff, truncation_psi):
        ones = np.ones_like(self.mixing_layer_indices, dtype=np.float32)
        if truncation_cutoff is None:
            truncation_coefs = ones * truncation_psi
        else:
            truncation_coefs = ones
            for index in range(self.n_broadcast):
                if index < truncation_cutoff:
                    truncation_coefs[:, index, :] = truncation_psi

        truncated_w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)
        return truncated_w_broadcasted

    @tf.function
    def call(self, inputs, truncation_cutoff=None, truncation_psi=1.0, training=None, mask=None):
        latents, labels = inputs

        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)

        if training:
            self.update_moving_average_of_w(w_broadcasted)
            w_broadcasted = self.style_mixing_regularization(latents, labels, w_broadcasted)

        if not training:
            w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff, truncation_psi)

        image_out = self.synthesis(w_broadcasted)
        return image_out, w_broadcasted

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        # shape_latents, shape_labels = input_shape
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]

    @tf.function
    def serve(self, latents, labels, truncation_psi):
        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)
        w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff=None, truncation_psi=truncation_psi)
        image_out = self.synthesis(w_broadcasted)

        image_out.set_shape([None, 3, self.resolutions[-1], self.resolutions[-1]])
        return image_out


def main():
    batch_size = 4
    g_params_with_label = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
        'truncation_psi': 0.5,
        'truncation_cutoff': None,
    }

    test_z = np.ones((batch_size, g_params_with_label['z_dim']), dtype=np.float32)
    test_y = np.ones((batch_size, g_params_with_label['labels_dim']), dtype=np.float32)

    generator = Generator(g_params_with_label)
    fake_images1, _ = generator([test_z, test_y], training=True)
    fake_images2, _ = generator([test_z, test_y], training=False)
    generator.summary()

    print(fake_images1.shape)

    print()
    for v in generator.variables:
        print('{}: {}'.format(v.name, v.shape))
    return


if __name__ == '__main__':
    main()
