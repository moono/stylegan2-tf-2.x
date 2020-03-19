import os
import argparse
import numpy as np
import tensorflow as tf

from stylegan2.generator import Generator
from train import filter_resolutions_featuremaps


def load_generator(args):
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
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
    }

    # create models
    print('Create models')
    g_clone = Generator(g_params, dynamic=True)

    # finalize model (build)
    test_latent = np.ones((4, g_params['z_dim']), dtype=np.float32)
    test_labels = np.ones((4, g_params['labels_dim']), dtype=np.float32)
    _, __ = g_clone([test_latent, test_labels], training=False)
    # _, __ = g_clone([test_latent, test_labels], training=True)

    # setup saving locations (object based savings)
    ckpt_dir = os.path.join(args['model_base_dir'], args['name'])
    ckpt = tf.train.Checkpoint(g_clone=g_clone)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)

    # try to restore
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
        _, __ = g_clone.predict([test_latent, test_labels])
        
    return g_clone


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', default='stylegan2-ffhq', type=str)
    parser.add_argument('--train_res', default=32, type=int)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--export_base_dir', default='./exports', type=str)
    parser.add_argument('--export_version', default=1, type=int)
    args = vars(parser.parse_args())

    g_clone = load_generator(args)

    # g_clone.save(
    #     filepath=os.path.join(args['export_base_dir'], args['name'], str(args['export_version'])),
    #     include_optimizer=False,
    #     save_format='tf')
    tf.saved_model.save(
        g_clone,
        os.path.join(args['export_base_dir'], args['name'], str(args['export_version'])),
        signatures=g_clone.serve.get_concrete_function(
            latents=tf.TensorSpec(shape=[None, 512], dtype=tf.float32),
            labels=tf.TensorSpec(shape=[None, 0], dtype=tf.float32),
            truncation_psi=tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    return


if __name__ == '__main__':
    main()
