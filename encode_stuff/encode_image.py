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
    def __init__(self, resolutions, featuremaps, vgg16_layer_names, image_size, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.vgg16_layer_names = vgg16_layer_names
        self.image_size = image_size

        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')
        self.perceptual_model = self.load_perceptual_network()

    def load_perceptual_network(self):
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(self.image_size, self.image_size, 3))

        # for l in vgg16.layers:
        #     print('{}: {}'.format(l.name, l.output_shape))

        vgg16_output_layers = [l.output for l in vgg16.layers if l.name in self.vgg16_layer_names]
        perceptual_model = Model(vgg16.input, vgg16_output_layers)

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
    def __init__(self, params):
        # set variables
        self.batch_size = 1
        self.optimizer = params['optimizer']
        self.n_train_step = params['n_train_step']
        self.lambda_percept = params['lambda_percept']
        self.lambda_mse = params['lambda_mse']
        self.w_broadcasted_shape = [self.batch_size, 18, 512]
        self.image_size = params['image_size']
        self.generator_ckpt_dir = params['generator_ckpt_dir']
        self.output_dir = params['output_dir']
        self.vgg16_layer_names = params['vgg16_layer_names']
        self.save_every = 100

        # prepare result dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # set image & models
        self.target_image = self.load_image(params['target_image_fn'], self.image_size)
        self.encoder_model, self.sample_image = self.load_encoder_model()

        # precompute target image perceptual features
        self.target_features = self.encoder_model.run_perceptual_model(self.target_image)

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
    def convert_image_to_uint8(fake_image):
        image = adjust_dynamic_range(fake_image, range_in=(-1.0, 1.0), range_out=(0.0, 255.0),
                                     out_dtype=tf.dtypes.float32)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = tf.cast(image, dtype=tf.dtypes.uint8)
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
        # test_latent = np.ones((1, g_params['z_dim']), dtype=np.float32)
        test_latent = np.random.normal(loc=0.0, scale=1.0, size=(1, g_params['z_dim']))
        test_labels = np.ones((1, g_params['labels_dim']), dtype=np.float32)
        _ = generator([test_latent, test_labels], training=False)

        # try to restore from g_clone
        ckpt = tf.train.Checkpoint(g_clone=generator)
        manager = tf.train.CheckpointManager(ckpt, self.generator_ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print('Restored from {}'.format(manager.latest_checkpoint))
        else:
            raise ValueError('Wrong checkpoint dir!!')

        # sample image
        sample_image, __ = generator([test_latent, test_labels], truncation_psi=0.5, training=False)
        self.save_image(sample_image, os.path.join(self.output_dir, 'current_generator_sample.png'))

        # build encoder model
        encoder_model = EncoderModel(resolutions, featuremaps, self.vgg16_layer_names, self.image_size)
        test_dlatent_plus = np.ones(self.w_broadcasted_shape, dtype=np.float32)
        _, __ = encoder_model(test_dlatent_plus)

        # copy weights from generator
        encoder_model.set_weights(generator.synthesis)
        _, __ = encoder_model(test_dlatent_plus)

        # freeze weights
        for layer in encoder_model.layers:
            layer.trainable = False

        return encoder_model, sample_image

    def perceptual_loss(self, y_true_list, y_pred_list, mse):
        loss = 0.0
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            loss += self.lambda_percept * mse(y_pred, y_true)
        return loss

    @tf.function
    def step(self):
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            tape.watch([self.w_broadcasted, self.target_image] + self.target_features)

            # forward pass
            fake_image, embeddings = self.encoder_model(self.w_broadcasted)

            # losses
            loss = self.perceptual_loss(self.target_features, embeddings, mse)
            loss += self.lambda_mse * mse(fake_image, self.target_image)

        t_vars = [self.w_broadcasted]
        gradients = tape.gradient(loss, t_vars)
        self.optimizer.apply_gradients(zip(gradients, t_vars))
        return loss

    def encode_image(self):
        def write_to_tensorboard(writer, w_broadcasted, image, step):
            # save to tensorboard
            with writer.as_default():
                for ii in range(18):
                    tf.summary.histogram('w+_{:02d}'.format(ii), w_broadcasted[0, ii, :], step=step)
                tf.summary.image('encoded', image, step=step)
            return

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.output_dir)
        write_to_tensorboard(train_summary_writer, self.w_broadcasted,
                             image=self.convert_image_to_uint8(self.sample_image),
                             step=0)

        for ts in range(1, self.n_train_step + 1):
            # optimize step
            loss_val = self.step()

            # save results
            if ts % self.save_every == 0:
                print('[step {:05d}/{:05d}]: {}'.format(ts, self.n_train_step, loss_val.numpy()))

                fake_image = self.encoder_model.run_synthesis_model(self.w_broadcasted)
                write_to_tensorboard(train_summary_writer, self.w_broadcasted,
                                     image=self.convert_image_to_uint8(fake_image),
                                     step=ts)

                # self.save_image(fake_image=fake_image,
                #                 out_fn=os.path.join(self.output_dir, 'encoded_at_step_{:04d}.png'.format(ts)))

        # lets restore with optimized embeddings
        final_image = self.encoder_model.run_synthesis_model(self.w_broadcasted)
        self.save_image(final_image, out_fn=os.path.join(self.output_dir, 'final_encoded.png'))
        np.save(os.path.join(self.output_dir, 'final_encoded.npy'), self.w_broadcasted.numpy())
        return


def main():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    encode_params = {
        'target_image_fn': os.path.join(abs_path, './00011.png'),
        'image_size': 256,
        'generator_ckpt_dir': os.path.join(abs_path, '../official-converted'),
        # 'generator_ckpt_dir': '/mnt/vision-nas/moono/trained_models/stylegan2-tf-2.x/stylegan2-ffhq',
        'output_dir': os.path.join(abs_path, './encode_results'),

        # # puzer config
        # 'optimizer': tf.keras.optimizers.SGD(1.0),
        # 'n_train_step': 1000,
        # 'lambda_percept': 1.0,
        # 'lambda_mse': 1.0,
        # 'vgg16_layer_names': ['block3_conv3'],

        # image2stylegan config
        'optimizer': tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        'n_train_step': 5000,
        'lambda_percept': 1.0,
        'lambda_mse': 1.0,
        'vgg16_layer_names': ['block1_conv1', 'block1_conv2', 'block3_conv2', 'block4_conv2'],
    }

    image_encoder = EncodeImage(encode_params)
    image_encoder.encode_image()
    return


if __name__ == '__main__':
    main()
