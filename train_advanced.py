import os
import time
import argparse
import numpy as np
import tensorflow as tf


from dataset_ffhq import get_ffhq_dataset
from stylegan2.utils import preprocess_fit_train_image, postprocess_images, merge_batch_images
from stylegan2.generator import Generator
from stylegan2.discriminator import Discriminator


class Trainer(object):
    def __init__(self, t_params, name):
        self.model_base_dir = t_params['model_base_dir']
        self.tfrecord_dir = t_params['tfrecord_dir']
        self.shuffle_buffer_size = t_params['shuffle_buffer_size']
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']
        self.g_opt = t_params['g_opt']
        self.d_opt = t_params['d_opt']
        self.batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.lazy_regularization = t_params['lazy_regularization']

        self.r1_gamma = 10.0
        # self.r2_gamma = 0.0
        self.max_steps = int(np.ceil(self.n_total_image / self.batch_size))
        self.out_res = self.g_params['resolutions'][-1]
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_reg: {:.3f}, pl_reg: {:.3f}'
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False

        # setup optimizer params
        self.g_opt = self.set_optimizer_params(self.g_opt)
        self.d_opt = self.set_optimizer_params(self.d_opt)
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False)
        self.pl_decay = 0.01
        self.pl_weight = 2.0

        # grab dataset
        print('Setting datasets')
        self.dataset = get_ffhq_dataset(self.tfrecord_dir, self.out_res, self.shuffle_buffer_size,
                                        self.batch_size, epochs=None)

        # create models
        print('Create models')
        self.generator = Generator(self.g_params)
        self.discriminator = Discriminator(self.d_params)
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])
        self.g_clone = Generator(self.g_params)

        # finalize model (build)
        test_latent = np.ones((1, self.g_params['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, self.g_params['labels_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, self.out_res, self.out_res), dtype=np.float32)
        _, __ = self.generator([test_latent, test_labels], training=False)
        _ = self.discriminator([test_images, test_labels], training=False)
        _, __ = self.g_clone([test_latent, test_labels], training=False)
        print('Copying g_clone')
        self.g_clone.set_weights(self.generator.get_weights())

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        generator=self.generator,
                                        g_clone=self.g_clone)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

        # try to restore
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('Restored from {}'.format(self.manager.latest_checkpoint))

            # check if already trained in this resolution
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return
        else:
            print('Not restoring from saved checkpoint')

    def set_optimizer_params(self, params):
        if self.lazy_regularization:
            mb_ratio = params['reg_interval'] / (params['reg_interval'] + 1)
            params['learning_rate'] = params['learning_rate'] * mb_ratio
            params['beta1'] = params['beta1'] ** mb_ratio
            params['beta2'] = params['beta2'] ** mb_ratio
        return params

    # @tf.function
    def d_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)
            d_loss = tf.reduce_mean(d_loss)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    # @tf.function
    def d_reg_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)

            # simple GP
            with tf.GradientTape() as p_tape:
                p_tape.watch(real_images)
                real_loss = tf.reduce_sum(self.discriminator([real_images, labels], training=True))

            real_grads = p_tape.gradient(real_loss, real_images)
            r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_penalty = tf.expand_dims(r1_penalty, axis=1)
            r1_penalty = r1_penalty * self.d_opt['reg_interval']

            # combine
            d_loss += r1_penalty * (0.5 * self.r1_gamma)
            d_loss = tf.reduce_mean(d_loss)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(r1_penalty)

    # @tf.function
    def g_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    # @tf.function
    def g_reg_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, w_broadcasted = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)

            # path length regularization
            h, w = fake_images.shape[2], fake_images.shape[3]

            # Compute |J*y|.
            with tf.GradientTape() as pl_tape:
                pl_tape.watch(w_broadcasted)
                pl_noise = tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) / np.sqrt(h * w)

            pl_grads = pl_tape.gradient(tf.reduce_sum(fake_images * pl_noise), w_broadcasted)
            pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

            # Track exponential moving average of |J*y|.
            pl_mean_val = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
            self.pl_mean.assign(pl_mean_val)

            # Calculate (|J*y|-a)^2.
            pl_penalty = tf.square(pl_lengths - self.pl_mean)

            # compute
            pl_reg = pl_penalty * self.pl_weight

            # combine
            g_loss += pl_reg
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, tf.reduce_mean(pl_reg)

    def train(self):
        if self.reached_max_steps:
            return

        # start actual training
        print('Start Training')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_reg = tf.keras.metrics.Mean('r1_reg', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        print('max_steps: {}'.format(self.max_steps))
        losses = {'g_loss': 0.0, 'd_loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0}
        t_start = time.time()

        for real_images in self.dataset:
            # preprocess inputs
            z = tf.random.normal(shape=[tf.shape(real_images)[0], self.g_params['z_dim']], dtype=tf.dtypes.float32)
            real_images = preprocess_fit_train_image(real_images, self.out_res)
            labels = tf.ones((tf.shape(real_images)[0], self.g_params['labels_dim']), dtype=tf.dtypes.float32)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # d train step
            if step % self.d_opt['reg_interval'] == 0:
                d_loss, r1_reg = self.d_reg_train_step(z, real_images, labels)

                # update values for printing
                losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = r1_reg.numpy()

                # update metrics
                metric_d_loss(d_loss)
                metric_r1_reg(r1_reg)
            else:
                d_loss = self.d_train_step(z, real_images, labels)

                # update values for printing
                losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = 0.0

                # update metrics
                metric_d_loss(d_loss)

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            # g train step
            if step % self.g_opt['reg_interval'] == 0:
                g_loss, pl_reg = self.g_reg_train_step(z, labels)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = pl_reg.numpy()

                # update metrics
                metric_g_loss(g_loss)
                metric_pl_reg(pl_reg)
            else:
                g_loss = self.g_train_step(z, labels)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = 0.0

                # update metrics
                metric_g_loss(g_loss)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_reg', metric_r1_reg.result(), step=step)
                tf.summary.scalar('pl_reg', metric_pl_reg.result(), step=step)
                tf.summary.histogram('w_avg', self.generator.w_avg, step=step)

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                summary_image = self.sample_images_tensorboard(real_images)
                with train_summary_writer.as_default():
                    tf.summary.image('images', summary_image, step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed,
                                               losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break

        # get current step
        step = self.g_optimizer.iterations.numpy()
        elapsed = time.time() - t_start
        print(self.log_template.format(step, elapsed,
                                       losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))

        # save last checkpoint
        self.manager.save(checkpoint_number=step)
        return

    def sample_images_tensorboard(self, real_images):
        # prepare inputs
        reals = real_images[:self.n_samples, :, :, :]
        latents = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
        dummy_labels = tf.ones((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)

        # run networks
        fake_images_00, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.0, training=False)
        fake_images_05, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.5, training=False)
        fake_images_07, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.7, training=False)
        fake_images_10, _ = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

        # merge on batch dimension: [5 * n_samples, 3, out_res, out_res]
        out = tf.concat([reals, fake_images_00, fake_images_05, fake_images_07, fake_images_10], axis=0)

        # prepare for image saving: [5 * n_samples, out_res, out_res, 3]
        out = postprocess_images(out)

        # resize to save disk spaces: [5 * n_samples, size, size, 3]
        size = min(self.out_res, 256)
        out = tf.image.resize(out, size=[size, size])

        # make single image and add batch dimension for tensorboard: [1, 5 * size, n_samples * size, 3]
        out = merge_batch_images(out, size, rows=5, cols=self.n_samples)
        out = np.expand_dims(out, axis=0)
        return out


def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--tfrecord_dir', default='/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq', type=str)
    parser.add_argument('--train_res', default=256, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    args = vars(parser.parse_args())

    # network params
    train_res = args['train_res']
    resolutions = [  4,   8,  16,  32,  64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128,  64,   32]
    train_resolutions, train_featuremaps = filter_resolutions_featuremaps(resolutions, featuremaps, train_res)
    g_params = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
    }
    d_params = {
        'labels_dim': 0,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }

    # training parameters
    training_parameters = {
        # global params
        **args,

        # network params
        'g_params': g_params,
        'd_params': d_params,

        # training params
        'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 4},
        'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 16},
        'batch_size': args['batch_size'],
        'n_total_image': 25000000,
        'n_samples': 4,
        'lazy_regularization': True,
    }

    trainer = Trainer(training_parameters, name='stylegan2-ffhq')
    trainer.train()
    return


if __name__ == '__main__':
    main()
