import tensorflow as tf


def load_generator(g_params=None, is_g_clone=False, ckpt_dir=None, custom_cuda=True):
    if custom_cuda:
        from stylegan2.generator import Generator
    else:
        from stylegan2_ref.generator import Generator

    if g_params is None:
        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        }

    test_latent = tf.ones((1, g_params['z_dim']), dtype=tf.float32)
    test_labels = tf.ones((1, g_params['labels_dim']), dtype=tf.float32)

    # build generator model
    generator = Generator(g_params)
    _ = generator([test_latent, test_labels])

    if ckpt_dir is not None:
        if is_g_clone:
            ckpt = tf.train.Checkpoint(g_clone=generator)
        else:
            ckpt = tf.train.Checkpoint(generator=generator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(f'Generator restored from {manager.latest_checkpoint}')
    return generator


def load_discriminator(d_params=None, ckpt_dir=None, custom_cuda=True):
    if custom_cuda:
        from stylegan2.discriminator import Discriminator
    else:
        from stylegan2_ref.discriminator import Discriminator

    if d_params is None:
        d_params = {
            'labels_dim': 0,
            'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        }

    res = d_params['resolutions'][-1]
    test_images = tf.ones((1, 3, res, res), dtype=tf.float32)
    test_labels = tf.ones((1, d_params['labels_dim']), dtype=tf.float32)

    # build discriminator model
    discriminator = Discriminator(d_params)
    _ = discriminator([test_images, test_labels])

    if ckpt_dir is not None:
        ckpt = tf.train.Checkpoint(discriminator=discriminator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print('Discriminator restored from {}'.format(manager.latest_checkpoint))
    return discriminator
