import os
import argparse
import tensorflow as tf

from load_models import load_generator


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', default='stylegan2-ffhq', type=str)
    parser.add_argument('--export_base_dir', default='./exports', type=str)
    parser.add_argument('--export_version', default=1, type=int)
    args = vars(parser.parse_args())

    # restore official converted model
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir='./official-converted')

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
