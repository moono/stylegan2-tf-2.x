import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from PIL import Image

from stylegan2.generator import Generator, Synthesis
from stylegan2.utils import adjust_dynamic_range


class EncoderModel(tf.keras.Model):
    def __init__(self, resolutions, featuremaps, image_size, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.image_size = image_size

        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')
        self.perceptual_model = self.load_perceptual_network()

    def load_perceptual_network(self):
        vgg16_layer_name = 'block3_conv3'  # 9: [None, 64, 64, 256]
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(self.image_size, self.image_size, 3))
        perceptual_model = Model(vgg16.input, vgg16.get_layer(vgg16_layer_name).output)

        # freeze weights
        for layer in perceptual_model.layers:
            layer.trainable = False
        return perceptual_model

    def set_weights(self, src_net):
        def split_first_name(name):
            splitted = name.split('/')
            new_name = '/'.join(splitted[1:])
            return new_name

        n_synthesis_weights = 0
        successful_copies = 0
        for cw in self.trainable_weights:
            if 'g_synthesis' in cw.name:
                n_synthesis_weights += 1

                cw_name = split_first_name(cw.name)
                for sw in src_net.trainable_weights:
                    sw_name = split_first_name(sw.name)
                    if cw_name == sw_name:
                        assert sw.shape == cw.shape
                        cw.assign(sw)
                        successful_copies += 1
                        break

        assert successful_copies == n_synthesis_weights
        return

    @staticmethod
    def post_process_image(image, image_size):
        # shrink = image.shape[-1] // image_size
        # ksize = [1, 1, shrink, shrink]
        # image = tf.nn.avg_pool(image, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

        image = adjust_dynamic_range(image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        # image = tf.cast(image, dtype=tf.dtypes.uint8)
        image = tf.image.resize(image, size=(image_size, image_size))

        image = vgg16_preprocess_input(image)
        return image

    def run_synthesis_model(self, embeddings):
        return self.synthesis(embeddings)

    def run_perceptual_model(self, image):
        return self.perceptual_model(image)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        fake_image = self.synthesis(x)
        fake_image = self.post_process_image(fake_image, self.image_size)
        embeddings = self.perceptual_model(fake_image)
        return fake_image, embeddings


class EncodeImage(object):
    def __init__(self, target_image_fn, image_size, generator_ckpt_dir):
        # set variables
        self.batch_size = 1
        self.learning_rate = 0.002
        self.n_train_step = 1000
        self.w_broadcasted_shape = [self.batch_size, 18, 512]
        self.image_size = image_size
        self.ckpt_dir = generator_ckpt_dir
        self.save_every = 100

        # set image & models
        self.target_image = self.load_image(target_image_fn, image_size)
        self.encoder_model = self.load_encoder_model()

        # precompute target image perceptual features
        self.target_feature = self.encoder_model.run_perceptual_model(self.target_image)

        # prepare optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # prepare variables to optimize
        self.w_broadcasted = tf.Variable(np.zeros(shape=self.w_broadcasted_shape, dtype=np.float32), trainable=True)
        return

    @staticmethod
    def load_image(image_fn, image_size):
        image = Image.open(image_fn)
        image = image.resize((image_size, image_size))
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        image = tf.constant(image, dtype=tf.dtypes.float32)
        image = vgg16_preprocess_input(image)
        return image

    @staticmethod
    def save_image(fake_image, out_fn):
        image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0),
                                     out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.cast(image, dtype=tf.dtypes.uint8)
        image = tf.squeeze(image, axis=0)
        image = Image.fromarray(image.numpy())
        image.save(out_fn)
        return

    def load_encoder_model(self):
        # build generator object
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]

        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': resolutions,
            'featuremaps': featuremaps,
            'w_ema_decay': 0.995,
            'style_mixing_prob': 0.9,
        }
        generator = Generator(g_params)
        test_latent = np.ones((1, g_params['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, g_params['labels_dim']), dtype=np.float32)
        _ = generator([test_latent, test_labels], training=False)

        # try to restore from g_clone
        ckpt = tf.train.Checkpoint(g_clone=generator)
        manager = tf.train.CheckpointManager(ckpt, self.ckpt_dir, max_to_keep=2)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print('Restored from {}'.format(manager.latest_checkpoint))
        else:
            raise ValueError('Wrong checkpoint dir!!')

        # build encoder model
        encoder_model = EncoderModel(resolutions, featuremaps, self.image_size)
        test_dlatent_plus = np.ones(self.w_broadcasted_shape, dtype=np.float32)
        _, __ = encoder_model(test_dlatent_plus)

        # copy weights from generator
        encoder_model.set_weights(generator.synthesis)
        _, __ = encoder_model(test_dlatent_plus)

        # freeze weights
        for layer in encoder_model.layers:
            layer.trainable = False

        return encoder_model

    # @tf.function
    def step(self):
        with tf.GradientTape() as tape:
            tape.watch([self.w_broadcasted, self.target_feature, self.target_image])

            # forward pass
            fake_image, embeddings = self.encoder_model(self.w_broadcasted)

            # losses
            loss = tf.reduce_mean(tf.square(embeddings - self.target_feature))
            loss += tf.reduce_mean(tf.abs(fake_image - self.target_image))

        t_vars = [self.w_broadcasted]
        gradients = tape.gradient(loss, t_vars)
        self.optimizer.apply_gradients(zip(gradients, t_vars))
        return loss

    def encode_image(self):
        for ts in range(self.n_train_step):
            loss_val = self.step()

            print('[step {:05d}/{:05d}]: {}'.format(ts, self.n_train_step, loss_val.numpy()))

            if ts % self.save_every == 0:
                fake_image = self.encoder_model.run_synthesis_model(self.w_broadcasted)
                self.save_image(fake_image, out_fn='encoded_at_step_{:04d}.png'.format(ts))

        # lets restore with optimized embeddings
        final_image = self.encoder_model.run_synthesis_model(self.w_broadcasted)
        self.save_image(final_image, out_fn='final_encoded.png')
        return


def main():
    image_fn = './00011.png'
    image_size = 256
    generator_ckpt_dir = './models/__stylegan2-ffhq'
    image_encoder = EncodeImage(image_fn, image_size, generator_ckpt_dir)
    image_encoder.encode_image()
    return


if __name__ == '__main__':
    main()
