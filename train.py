import os
import time
import argparse
import numpy as np
import tensorflow as tf

from tf_utils.utils import allow_memory_growth
from dataset_ffhq import get_ffhq_dataset
from stylegan2.utils import preprocess_fit_train_image, postprocess_images, merge_batch_images
from load_models import load_generator, load_discriminator


def initiate_models(g_params, d_params):
    discriminator = load_discriminator(d_params, ckpt_dir=None)
    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=None)
    g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None)

    # set initial g_clone weights same as generator
    g_clone.set_weights(generator.get_weights())
    return discriminator, generator, g_clone


class Trainer(object):
    def __init__(self, t_params, name):
        self.model_base_dir = t_params['model_base_dir']
        self.global_batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.max_steps = int(np.ceil(self.n_total_image / self.global_batch_size))
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.train_res = t_params['train_res']
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_penalty: {:.3f}'

        self.r1_gamma = 10.0
        self.g_opt = t_params['g_opt']
        self.d_opt = t_params['d_opt']
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']

        # create model: model and optimizer must be created under `strategy.scope`
        self.discriminator, self.generator, self.g_clone = initiate_models(self.g_params, self.d_params)

        # set optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])

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

    def d_train_step(self, dist_inputs):
        z, real_images, labels = dist_inputs

        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images = self.generator([z, labels], training=True)
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
            r1_pen = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_pen = tf.expand_dims(r1_pen, axis=1)

            # scale losses
            d_loss = tf.reduce_sum(d_loss) * (1.0 / self.global_batch_size)
            r1_pen = tf.reduce_sum(r1_pen) * (1.0 / self.global_batch_size)

            # combine
            d_loss += r1_pen * (0.5 * self.r1_gamma)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, r1_pen

    def g_train_step(self, dist_inputs):
        z, labels = dist_inputs

        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)

            # scale losses
            g_loss = tf.reduce_sum(g_loss) * (1.0 / self.global_batch_size)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    def train(self, dist_datasets, strategy):
        def dist_d_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_r1_pen = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            return mean_d_loss, mean_r1_pen

        def dist_g_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            return mean_g_loss

        # wrap with tf.function
        if True:
            dist_d_train_step = tf.function(dist_d_train_step)
            dist_g_train_step = tf.function(dist_g_train_step)

        if self.reached_max_steps:
            return

        # start actual training
        print('Start Training')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_pen = tf.keras.metrics.Mean('r1_penalty', dtype=tf.float32)

        # start training
        print('max_steps: {}'.format(self.max_steps))
        t_start = time.time()

        for real_images in dist_datasets:
            # preprocess inputs
            batch_size = tf.shape(real_images)[0]
            z = tf.random.normal(shape=[batch_size, self.g_params['z_dim']], dtype=tf.dtypes.float32)
            labels = tf.ones(shape=[batch_size, self.g_params['labels_dim']], dtype=tf.dtypes.float32)
            real_images = preprocess_fit_train_image(real_images, self.train_res)

            # train step
            d_loss, r1_pen = dist_d_train_step((z, real_images, labels))
            self.g_clone.set_as_moving_average_of(self.generator)
            g_loss = dist_g_train_step((z, labels))

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_r1_pen(r1_pen)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_penalty', metric_r1_pen.result(), step=step)

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
                print(self.log_template.format(step, elapsed, d_loss.numpy(), g_loss.numpy(), r1_pen.numpy()))

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.g_optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return

    def sample_images_tensorboard(self, real_images):
        # prepare inputs
        reals = real_images[:self.n_samples, :, :, :]
        latents = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
        dummy_labels = tf.ones((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)

        # run networks
        fake_images_00 = self.g_clone([latents, dummy_labels], truncation_psi=0.0, training=False)
        fake_images_05 = self.g_clone([latents, dummy_labels], truncation_psi=0.5, training=False)
        fake_images_07 = self.g_clone([latents, dummy_labels], truncation_psi=0.7, training=False)
        fake_images_10 = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

        # merge on batch dimension: [5 * n_samples, 3, out_res, out_res]
        out = tf.concat([reals, fake_images_00, fake_images_05, fake_images_07, fake_images_10], axis=0)

        # prepare for image saving: [5 * n_samples, out_res, out_res, 3]
        out = postprocess_images(out)

        # resize to save disk spaces: [5 * n_samples, size, size, 3]
        if self.train_res > 256:
            size = 256
        else:
            size = self.train_res
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
    parser.add_argument('--tfrecord_dir', default='./tfrecords', type=str)
    parser.add_argument('--train_res', default=64, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size_per_replica', default=4, type=int)
    args = vars(parser.parse_args())

    # allow_memory_growth()

    # network params
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    train_resolutions, train_featuremaps = filter_resolutions_featuremaps(resolutions, featuremaps, args['train_res'])
    g_params = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }
    d_params = {
        'labels_dim': 0,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }

    # prepare distribute strategy
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # prepare dataset
    dataset = get_ffhq_dataset(args['tfrecord_dir'], args['train_res'], batch_size=global_batch_size, epochs=None)

    with strategy.scope():
        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # training parameters
        training_parameters = {
            # global params
            'model_base_dir': args['model_base_dir'],

            # network params
            'g_params': g_params,
            'd_params': d_params,

            # training params
            'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
            'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
            'batch_size': global_batch_size,
            'n_total_image': 25000000,
            'n_samples': 3,
            'train_res': args['train_res'],
        }

        trainer = Trainer(training_parameters, name=f'stylegan2-ffhq-{args["train_res"]}x{args["train_res"]}')
        trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
