import os
import time
import argparse
import numpy as np
import tensorflow as tf

from utils import str_to_bool
from tf_utils import check_tf_version, allow_memory_growth, split_gpu_for_testing
from load_models import load_generator, load_discriminator
from dataset_ffhq import get_ffhq_dataset
from losses import d_logistic, d_logistic_r1_reg, g_logistic_non_saturating, g_logistic_ns_pathreg


def initiate_models(g_params, d_params, use_custom_cuda):
    discriminator = load_discriminator(d_params, ckpt_dir=None, custom_cuda=use_custom_cuda)
    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=None, custom_cuda=use_custom_cuda)
    g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=use_custom_cuda)

    # set initial g_clone weights same as generator
    g_clone.set_weights(generator.get_weights())
    return discriminator, generator, g_clone


class Trainer(object):
    def __init__(self, t_params, name):
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.use_tf_function = t_params['use_tf_function']
        self.use_custom_cuda = t_params['use_custom_cuda']
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
        self.log_template = '{:s}, {:s}, {:s}'.format(
            'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}',
            'd_gan_loss: {:.3f}, g_gan_loss: {:.3f}',
            'r1_penalty: {:.3f}, pl_penalty: {:.3f}')

        # copy network params
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']

        # set optimizer params
        self.global_batch_scaler = 1.0 / float(self.global_batch_size)
        self.r1_gamma = 10.0
        self.g_opt = self.update_optimizer_params(t_params['g_opt'])
        self.d_opt = self.update_optimizer_params(t_params['d_opt'])
        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_denorm = tf.math.rsqrt(float(self.train_res) * float(self.train_res))
        self.pl_decay = 0.01
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False,
                                   synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        # create model: model and optimizer must be created under `strategy.scope`
        self.discriminator, self.generator, self.g_clone = initiate_models(self.g_params,
                                                                           self.d_params,
                                                                           self.use_custom_cuda)

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

    @staticmethod
    def update_optimizer_params(params):
        params_copy = params.copy()
        mb_ratio = params_copy['reg_interval'] / (params_copy['reg_interval'] + 1)
        params_copy['learning_rate'] = params_copy['learning_rate'] * mb_ratio
        params_copy['beta1'] = params_copy['beta1'] ** mb_ratio
        params_copy['beta2'] = params_copy['beta2'] ** mb_ratio
        return params_copy

    def d_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_loss = d_logistic(real_images, self.generator, self.discriminator, self.g_params['z_dim'])

            # scale loss
            d_loss = tf.reduce_sum(d_loss) * self.global_batch_scaler

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    def d_train_step_reg(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_gan_loss, r1_penalty = d_logistic_r1_reg(real_images, self.generator, self.discriminator,
                                                       self.g_params['z_dim'])
            r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_opt['reg_interval']

            # scale losses
            r1_penalty = tf.reduce_sum(r1_penalty) * self.global_batch_scaler
            d_gan_loss = tf.reduce_sum(d_gan_loss) * self.global_batch_scaler

            # combine
            d_loss = d_gan_loss + r1_penalty

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, d_gan_loss, r1_penalty

    def g_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_loss = g_logistic_non_saturating(real_images, self.generator, self.discriminator, self.g_params['z_dim'])

            # scale loss
            g_loss = tf.reduce_sum(g_loss) * self.global_batch_scaler

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    def g_train_step_reg(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pl_penalty = g_logistic_ns_pathreg(real_images, self.generator, self.discriminator,
                                                           self.g_params['z_dim'], self.pl_mean,
                                                           self.pl_minibatch_shrink, self.pl_denorm, self.pl_decay)
            pl_penalty = pl_penalty * self.pl_weight * self.g_opt['reg_interval']

            # scale loss
            pl_penalty = tf.reduce_sum(pl_penalty) * self.global_batch_scaler
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.global_batch_scaler

            # combine
            g_loss = g_gan_loss + pl_penalty

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, g_gan_loss, pl_penalty

    def train(self, dist_datasets, strategy):
        def dist_d_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.d_train_step, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_d_loss

        def dist_d_train_step_reg(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step_reg, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.d_train_step_reg, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_d_gan_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            mean_r1_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
            return mean_d_loss, mean_d_gan_loss, mean_r1_penalty

        def dist_g_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.g_train_step, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_g_loss

        def dist_g_train_step_reg(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step_reg, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.g_train_step_reg, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_g_gan_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            mean_pl_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
            return mean_g_loss, mean_g_gan_loss, mean_pl_penalty

        def dist_gen_samples(dist_inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_samples = strategy.experimental_run_v2(self.gen_samples, args=(dist_inputs,))
            else:
                per_replica_samples = strategy.run(self.gen_samples, args=(dist_inputs,))
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
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_gan_loss = tf.keras.metrics.Mean('d_gan_loss', dtype=tf.float32)
        metric_g_gan_loss = tf.keras.metrics.Mean('g_gan_loss', dtype=tf.float32)
        metric_r1_penalty = tf.keras.metrics.Mean('r1_penalty', dtype=tf.float32)
        metric_pl_penalty = tf.keras.metrics.Mean('pl_penalty', dtype=tf.float32)

        # start training
        zero = tf.constant(0.0, dtype=tf.float32)
        print('max_steps: {}'.format(self.max_steps))
        t_start = time.time()
        for real_images in dist_datasets:
            step = self.g_optimizer.iterations.numpy()

            # d train step
            if (step + 1) % self.d_opt['reg_interval'] == 0:
                d_loss, d_gan_loss, r1_penalty = dist_d_train_step_reg((real_images, ))
            else:
                d_loss = dist_d_train_step((real_images, ))
                d_gan_loss = d_loss
                r1_penalty = zero

            # g train step
            if (step + 1) % self.g_opt['reg_interval'] == 0:
                g_loss, g_gan_loss, pl_penalty = dist_g_train_step_reg((real_images, ))
            else:
                g_loss = dist_g_train_step((real_images,))
                g_gan_loss = g_loss
                pl_penalty = zero

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_d_gan_loss(d_gan_loss)
            metric_g_gan_loss(g_gan_loss)
            metric_r1_penalty(r1_penalty)
            metric_pl_penalty(pl_penalty)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_gan_loss', metric_d_gan_loss.result(), step=step)
                tf.summary.scalar('g_gan_loss', metric_g_gan_loss.result(), step=step)
                tf.summary.scalar('r1_penalty', metric_r1_penalty.result(), step=step)
                tf.summary.scalar('pl_penalty', metric_pl_penalty.result(), step=step)

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
                                               d_gan_loss.numpy(), g_gan_loss.numpy(),
                                               r1_penalty.numpy(), pl_penalty.numpy()))

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
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
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
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_custom_cuda', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--tfrecord_dir', default='./tfrecords', type=str)
    parser.add_argument('--train_res', default=256, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size_per_replica', default=4, type=int)
    args = vars(parser.parse_args())

    # check tensorflow version
    cur_tf_ver = check_tf_version()

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing(mem_in_gb=4.5)

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
            'cur_tf_ver': cur_tf_ver,
            'use_tf_function': args['use_tf_function'],
            'use_custom_cuda': args['use_custom_cuda'],
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
        }

        trainer = Trainer(training_parameters, name=f'stylegan2-ffhq-{args["train_res"]}x{args["train_res"]}')
        trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
