import os
import time
import argparse
import numpy as np
import tensorflow as tf

from utils import str_to_bool
from tf_utils import allow_memory_growth
from dataset_ffhq import get_ffhq_dataset
from stylegan2.losses import d_logistic, d_logistic_r1, g_logistic_non_saturating, g_logistic_ns_pathreg
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
        self.use_tf_function = t_params['use_tf_function']
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
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_pen: {:.3f}, pl_reg: {:.3f}'

        # copy network params
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']

        # set optimizer params
        self.lazy_regularization = t_params['lazy_regularization']
        self.r1_gamma = 10.0
        self.g_opt = self.update_optimizer_params(t_params['g_opt'])
        self.d_opt = self.update_optimizer_params(t_params['d_opt'])
        self.pl_decay = 0.01
        self.pl_weight = 1.0
        self.pl_denorm = 1.0 / np.sqrt(self.train_res * self.train_res)
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False,
                                   synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

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
                                        g_clone=self.g_clone,
                                        pl_mean=self.pl_mean)
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

    def update_optimizer_params(self, params):
        params_copy = params.copy()
        if self.lazy_regularization:
            mb_ratio = params_copy['reg_interval'] / (params_copy['reg_interval'] + 1)
            params_copy['learning_rate'] = params_copy['learning_rate'] * mb_ratio
            params_copy['beta1'] = params_copy['beta1'] ** mb_ratio
            params_copy['beta2'] = params_copy['beta2'] ** mb_ratio
        return params_copy

    def d_train_step(self, dist_inputs):
        batch_size, real_images = dist_inputs

        with tf.GradientTape() as d_tape:
            # compute losses
            d_loss = d_logistic(self.generator, self.discriminator, real_images, batch_size,
                                self.g_params['z_dim'], self.g_params['labels_dim'])

            # scale losses
            d_loss = tf.reduce_sum(d_loss) * (1.0 / self.global_batch_size)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    def d_train_step_reg(self, dist_inputs):
        batch_size, real_images = dist_inputs

        with tf.GradientTape() as d_tape:
            # compute losses
            d_loss, r1_pen = d_logistic_r1(self.generator, self.discriminator, real_images, batch_size,
                                           self.g_params['z_dim'], self.g_params['labels_dim'])

            # scale losses
            d_loss = tf.reduce_sum(d_loss) * (1.0 / self.global_batch_size)
            r1_pen = tf.reduce_sum(r1_pen) * (1.0 / self.global_batch_size) * self.d_opt['reg_interval']

            # combine
            d_loss += r1_pen * (0.5 * self.r1_gamma)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, r1_pen

    def g_train_step(self, dist_inputs):
        batch_size = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_loss = g_logistic_non_saturating(self.generator, self.discriminator, batch_size,
                                               self.g_params['z_dim'], self.g_params['labels_dim'])

            # scale losses
            g_loss = tf.reduce_sum(g_loss) * (1.0 / self.global_batch_size)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    def g_train_step_reg(self, dist_inputs):
        batch_size = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_loss, pl_pen = g_logistic_ns_pathreg(self.generator, self.discriminator, batch_size,
                                                   self.g_params['z_dim'], self.g_params['labels_dim'],
                                                   self.pl_mean, pl_minibatch_shrink=2, pl_decay=self.pl_decay)

            # scale losses
            g_loss = tf.reduce_sum(g_loss) * (1.0 / self.global_batch_size)
            pl_pen = tf.reduce_sum(pl_pen) * (1.0 / self.global_batch_size) * self.g_opt['reg_interval']

            # combine
            g_loss += pl_pen * self.pl_weight

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, pl_pen

    def train(self, dist_datasets, strategy):
        def dist_d_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_d_loss

        def dist_d_train_step_reg(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step_reg, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_r1_pen = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            return mean_d_loss, mean_r1_pen

        def dist_g_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_g_loss

        def dist_g_train_step_reg(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step_reg, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_pl_pen = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            return mean_g_loss, mean_pl_pen

        def dist_gen_samples(dist_inputs):
            # `experimental_run_v2` replicates the provided computation and runs it with the distributed input.
            per_replica_samples = strategy.experimental_run_v2(self.gen_samples, args=(dist_inputs,))
            return per_replica_samples

        # wrap with tf.function
        if self.use_tf_function:
            dist_d_train_step = tf.function(dist_d_train_step)
            dist_g_train_step = tf.function(dist_g_train_step)
            dist_d_train_step_reg = tf.function(dist_d_train_step_reg)
            dist_g_train_step_reg = tf.function(dist_g_train_step_reg)
            dist_gen_samples = tf.function(dist_gen_samples)

        if self.reached_max_steps:
            return

        # start actual training
        print('Start Training')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_pen = tf.keras.metrics.Mean('r1_pen', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        zero = tf.constant(0.0, dtype=tf.float32)
        print('max_steps: {}'.format(self.max_steps))
        t_start = time.time()
        for real_images in dist_datasets:
            # get current step
            step = self.g_optimizer.iterations.numpy()

            # train steps
            batch_size = tf.shape(real_images)[0]

            # g train step
            g_loss, pl_reg = dist_g_train_step((batch_size,)), zero
            if step % self.g_opt['reg_interval'] == 0:
                g_loss, pl_reg = dist_g_train_step_reg((batch_size,))

            # d train step
            d_loss, r1_pen = dist_d_train_step((batch_size, real_images)), zero
            if step % self.d_opt['reg_interval'] == 0:
                d_loss, r1_pen = dist_d_train_step_reg((batch_size, real_images))

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_r1_pen(r1_pen)
            metric_pl_reg(pl_reg)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_pen', metric_r1_pen.result(), step=step)
                tf.summary.scalar('pl_reg', metric_pl_reg.result(), step=step)

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                test_z = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
                test_labels = tf.ones((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)
                summary_image = dist_gen_samples((test_z, test_labels))

                # convert to tensor image
                summary_image = self.convert_per_replica_image(summary_image, strategy)

                with train_summary_writer.as_default():
                    tf.summary.image('images', summary_image, step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed, d_loss.numpy(), g_loss.numpy(),
                                               r1_pen.numpy(), pl_reg.numpy()))

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.g_optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return

    def gen_samples(self, inputs):
        test_z, test_labels = inputs

        # run networks
        fake_images_05 = self.g_clone([test_z, test_labels], truncation_psi=0.5, training=False)
        fake_images_07 = self.g_clone([test_z, test_labels], truncation_psi=0.7, training=False)

        # merge on batch dimension: [n_samples, 3, out_res, 2 * out_res]
        final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return final_image

    @staticmethod
    def convert_per_replica_image(nchw_per_replica_images, strategy):
        as_tensor = tf.concat(strategy.experimental_local_results(nchw_per_replica_images), axis=0)
        as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
        as_tensor = tf.clip_by_value(as_tensor, 0.0, 1.0) * 255.0
        as_tensor = tf.cast(as_tensor, tf.uint8)
        return as_tensor


def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--tfrecord_dir', default='./tfrecords', type=str)
    parser.add_argument('--train_res', default=64, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size_per_replica', default=2, type=int)
    args = vars(parser.parse_args())

    if args['allow_memory_growth']:
        allow_memory_growth()

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
            'use_tf_function': args['use_tf_function'],
            'model_base_dir': args['model_base_dir'],

            # network params
            'g_params': g_params,
            'd_params': d_params,

            # training params
            'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 8},
            'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 16},
            'batch_size': global_batch_size,
            'n_total_image': 25000000,
            'n_samples': 3,
            'train_res': args['train_res'],
            'lazy_regularization': True,
        }

        trainer = Trainer(training_parameters, name=f'stylegan2-ffhq-{args["train_res"]}x{args["train_res"]}')
        trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
