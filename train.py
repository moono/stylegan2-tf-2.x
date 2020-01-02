import os
import glob
import time
import argparse
import numpy as np
import tensorflow as tf


from datasets.dataset_kun_f_42params_20000_tfrecord import input_fn
from commons.utils import preprocess_fit_train_image, postprocess_images, merge_batch_images
from network_v21.renderer import Renderer
from network_v21.discriminator import Discriminator


class Trainer(object):
    def __init__(self, t_params, name):
        self.model_base_dir = t_params['model_base_dir']
        self.tfrecord_dir = t_params['tfrecord_dir']
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']
        self.g_opt = t_params['g_opt']
        self.d_opt = t_params['d_opt']
        self.batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])

        self.r1_gamma = 10.0
        self.r2_gamma = 0.0
        self.max_steps = int(np.ceil(self.n_total_image / self.batch_size))
        self.reached_max_steps = False
        self.out_res = self.g_params['resolutions'][-1]
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_penalty: {:.3f}'
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100

        # grab dataset
        print('Setting datasets')
        self.train_ds, self.val_ds = self.get_dataset(self.out_res, self.batch_size, self.n_samples)
        self.val_ds_iter = iter(self.val_ds)

        # create models
        print('Create models')
        self.renderer = Renderer(self.g_params)
        self.discriminator = Discriminator(self.d_params)
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])
        self.r_clone = Renderer(self.g_params)

        # finalize model (build)
        test_x = np.ones((1, self.g_params['x_dim']), dtype=np.float32)
        test_labels = np.ones((1, self.g_params['x_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, self.out_res, self.out_res), dtype=np.float32)
        _ = self.renderer(test_x, training=False)
        _ = self.discriminator([test_images, test_labels], training=False)
        _ = self.r_clone(test_x, training=False)
        print('Copying g_clone')
        self.r_clone.set_weights(self.renderer.get_weights())

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        renderer=self.renderer,
                                        r_clone=self.r_clone)
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

    def get_dataset(self, res, batch_size, n_samples):
        # get dataset
        filenames = glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecord'))
        filenames = sorted(filenames)
        n_train_tfrecord_files = len(filenames) - 1
        train_filenames = filenames[:n_train_tfrecord_files]
        val_filenames = [filenames[-1]]

        label_min_max = np.load(os.path.join(self.tfrecord_dir, 'label_min_max.npy'))
        label_max = label_min_max[1, :].tolist()
        label_depth = [x + 1 for x in label_max]

        train_ds = input_fn(train_filenames, res, label_depth, batch_size=batch_size, epochs=None)
        val_ds = input_fn(val_filenames, res, label_depth, batch_size=n_samples, epochs=None)
        return train_ds, val_ds

    @tf.function
    def d_train_step(self, x, real_images):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images = self.renderer(x, training=True)
            real_scores = self.discriminator([real_images, x], training=True)
            fake_scores = self.discriminator([fake_images, x], training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)

            # simple GP
            with tf.GradientTape() as p_tape:
                p_tape.watch(real_images)
                real_loss = tf.reduce_sum(self.discriminator([real_images, x], training=True))

            real_grads = p_tape.gradient(real_loss, real_images)
            r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_penalty = tf.expand_dims(r1_penalty, axis=1)

            # combine
            d_loss += r1_penalty * (0.5 * self.r1_gamma)
            d_loss = tf.reduce_mean(d_loss)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(r1_penalty)

    @tf.function
    def g_train_step(self, x):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images = self.renderer(x, training=True)
            fake_scores = self.discriminator([fake_images, x], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.renderer.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.renderer.trainable_variables))
        return g_loss

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
        metric_r1_pen = tf.keras.metrics.Mean('r1_penalty', dtype=tf.float32)

        # start training
        print('max_steps: {}'.format(self.max_steps))
        t_start = time.time()

        for real_images, x in self.train_ds:
            # preprocess inputs
            real_images = preprocess_fit_train_image(real_images, self.out_res)

            # train step
            d_loss, r1_penalty = self.d_train_step(x, real_images)
            self.r_clone.set_as_moving_average_of(self.renderer)
            g_loss = self.g_train_step(x)

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_r1_pen(r1_penalty)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_penalty', metric_r1_pen.result(), step=step)
                tf.summary.histogram('w_avg', self.renderer.w_avg, step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed, d_loss.numpy(), g_loss.numpy(), r1_penalty.numpy()))

                # reset timer
                t_start = time.time()

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                sample_images = self.sample_image(x, real_images)
                with train_summary_writer.as_default():
                    tf.summary.image('train_real_images', sample_images['train']['real'], step=step)
                    tf.summary.image('train_fake_images_eval', sample_images['train']['fake_eval'], step=step)
                    tf.summary.image('train_fake_images_clone', sample_images['train']['fake_clone'], step=step)
                    tf.summary.image('val_real_images', sample_images['val']['real'], step=step)
                    tf.summary.image('val_fake_images_eval', sample_images['val']['fake_eval'], step=step)
                    tf.summary.image('val_fake_images_clone', sample_images['val']['fake_clone'], step=step)

            # check exit status
            if step >= self.max_steps:
                break

        # get current step
        step = self.g_optimizer.iterations.numpy()
        elapsed = time.time() - t_start
        print(self.log_template.format(step, elapsed, d_loss.numpy(), g_loss.numpy(), r1_penalty.numpy()))

        # save last checkpoint
        self.manager.save(checkpoint_number=step)
        return

    def sample_image(self, train_x, train_real_images):
        def finalize_image(img, res, n_samples):
            img = postprocess_images(img)
            img = img.numpy()
            img = merge_batch_images(img, res, rows=1, cols=n_samples)
            img = np.expand_dims(img, axis=0)
            return img

        def finalize_image2(img, res, n_samples):
            img = tf.transpose(img, [0, 2, 3, 1])
            img = tf.cast(img, dtype=tf.dtypes.uint8)
            img = img.numpy()
            img = merge_batch_images(img, res, rows=1, cols=n_samples)
            img = np.expand_dims(img, axis=0)
            return img

        # deal with real images
        train_real_images_cropped = train_real_images[:self.n_samples, :, :, :]
        train_x_cropped = train_x[:self.n_samples, :]

        # grab validation sample real images too
        val_real_images_cropped, val_x_cropped = next(self.val_ds_iter)

        # run networks
        train_fake_images_eval = self.renderer(train_x_cropped, training=False)
        train_fake_images_clone = self.r_clone(train_x_cropped, training=False)
        val_fake_images_eval = self.renderer(val_x_cropped, training=False)
        val_fake_images_clone = self.r_clone(val_x_cropped, training=False)

        # convert to numpy image array
        t_real_img = finalize_image(train_real_images_cropped, self.out_res, self.n_samples)
        t_fake_img_eval = finalize_image(train_fake_images_eval, self.out_res, self.n_samples)
        t_fake_img_clone = finalize_image(train_fake_images_clone, self.out_res, self.n_samples)
        v_real_img = finalize_image2(val_real_images_cropped, self.out_res, self.n_samples)
        v_fake_img_eval = finalize_image(val_fake_images_eval, self.out_res, self.n_samples)
        v_fake_img_clone = finalize_image(val_fake_images_clone, self.out_res, self.n_samples)

        return {
            'train': {'real': t_real_img, 'fake_eval': t_fake_img_eval, 'fake_clone': t_fake_img_clone},
            'val': {'real': v_real_img, 'fake_eval': v_fake_img_eval, 'fake_clone': v_fake_img_clone},
        }


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_base_dir', default='../models', type=str)
    parser.add_argument('--tfrecord_dir', default='/mnt/my_data/image_data/b&s_face/kun_f_42params_20000/tfrecords', type=str)
    args = vars(parser.parse_args())

    label_min_max = np.load(os.path.join(args['tfrecord_dir'], 'label_min_max.npy'))
    label_max = label_min_max[1, :].tolist()
    label_depth = [x + 1 for x in label_max]

    # network params
    x_dim = 250
    resolutions = [4, 8, 16, 32, 64, 128, 256]
    featuremaps = [512, 512, 512, 512, 512, 256, 128]
    g_params = {
        'x_dim': x_dim,
        'w_dim': 512,
        'x_depth': label_depth,
        'n_mapping': 8,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
        'truncation_psi': 0.5,
        'truncation_cutoff': None,
    }
    d_params = {
        'labels_dim': x_dim,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
    }

    # training parameters
    training_parameters = {
        # global params
        **args,

        # network params
        'g_params': g_params,
        'd_params': d_params,

        # training params
        'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
        'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
        'batch_size': 4,
        'n_total_image': 25000000,
        'n_samples': 2,
    }

    trainer = Trainer(training_parameters, name='network_v21')
    trainer.train()
    return


if __name__ == '__main__':
    main()
