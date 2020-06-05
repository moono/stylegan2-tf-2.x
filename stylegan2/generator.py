import tensorflow as tf

from stylegan2.utils import lerp
from stylegan2.layers.mapping_block import Mapping
from stylegan2.layers.synthesis_block import Synthesis


class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.z_dim = g_params['z_dim']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.n_mapping = g_params['n_mapping']
        self.resolutions = g_params['resolutions']
        self.featuremaps = g_params['featuremaps']
        self.w_ema_decay = 0.995
        self.style_mixing_prob = 0.9

        self.n_broadcast = len(self.resolutions) * 2
        self.mixing_layer_indices = tf.range(self.n_broadcast, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]

        self.g_mapping = Mapping(self.w_dim, self.labels_dim, self.n_mapping, name='g_mapping')
        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, tf.newaxis], [1, self.n_broadcast, 1]))
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

    def truncation_trick(self, w_broadcasted, truncation_psi, truncation_cutoff=None):
        ones = tf.ones_like(self.mixing_layer_indices, dtype=tf.float32)
        tpsi = ones * truncation_psi
        if truncation_cutoff is None:
            truncation_coefs = tpsi
        else:
            indices = tf.range(self.n_broadcast)
            truncation_coefs = tf.where(condition=tf.less(indices, truncation_cutoff), x=tpsi, y=ones)

        truncated_w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)
        return truncated_w_broadcasted

    def call(self, inputs, truncation_psi=1.0, truncation_cutoff=None, training=None, mask=None):
        latents, labels = inputs

        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)

        if training:
            self.update_moving_average_of_w(w_broadcasted)
            w_broadcasted = self.style_mixing_regularization(latents, labels, w_broadcasted)

        if not training:
            w_broadcasted = self.truncation_trick(w_broadcasted, truncation_psi, truncation_cutoff)

        image_out = self.synthesis(w_broadcasted)
        return image_out, w_broadcasted

    def compute_output_shape(self, input_shape):
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
    batch_size = 1
    g_params_with_label = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
    }

    test_z = tf.ones((batch_size, g_params_with_label['z_dim']), dtype=tf.float32)
    test_y = tf.ones((batch_size, g_params_with_label['labels_dim']), dtype=tf.float32)

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
