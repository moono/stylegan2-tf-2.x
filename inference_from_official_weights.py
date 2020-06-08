# ##########################################################################################################
# To extract official weight from official repo(https://github.com/NVlabs/stylegan2), use the following code
# ##########################################################################################################
# ##########################################################################################################
# import tensorflow as tf
# import pretrained_networks
# 
# 
# network_pkl = './official_pretrained/stylegan2-ffhq-config-f.pkl'
# print('Loading networks from "%s"...' % network_pkl)
# _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
# print()
# 
# # Print network details.
# print('===_G===')
# _G.print_layers()
# print('===Gs===')
# Gs.print_layers()
# 
# t_var = tf.trainable_variables()
# pprint.pprint(t_var)
# 
# saver = tf.train.Saver()
# 
# save_path = saver.save(tf.get_default_session(), "./model.ckpt")
# ##########################################################################################################


import numpy as np
import tensorflow as tf

from PIL import Image
from stylegan2.utils import postprocess_images
from load_models import load_generator
from tf_utils import allow_memory_growth


def handle_mapping(w_name):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_mapping')
        indicator = splitted[index + 1]
        val = indicator.split('_')[-1]
        return val

    level = extract_info(w_name)
    if 'w' in w_name:
        official_var_name = 'G_mapping_1/Dense{}/weight'.format(level)
    else:
        official_var_name = 'G_mapping_1/Dense{}/bias'.format(level)
    return official_var_name


def handle_synthesis(w_name):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_synthesis')
        indicator1 = splitted[index + 1]
        indicator2 = splitted[index + 2]
        r = indicator1.split('x')[1]
        d = indicator2
        return r, d

    def to_rgb_layer(name, r):
        if 'conv/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/weight'.format(r, r)
        elif 'mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/mod_weight'.format(r, r)
        elif 'mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/mod_bias'.format(r, r)
        else:   # if 'bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/bias'.format(r, r)
        return o_name

    def handle_block_layer(name, r):
        if 'conv_0/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/weight'.format(r, r)
        elif 'conv_0/mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/mod_weight'.format(r, r)
        elif 'conv_0/mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/mod_bias'.format(r, r)
        elif 'noise_0/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/noise_strength'.format(r, r)
        elif 'bias_0/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/bias'.format(r, r)
        elif 'conv_1/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/weight'.format(r, r)
        elif 'conv_1/mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/mod_weight'.format(r, r)
        elif 'conv_1/mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/mod_bias'.format(r, r)
        elif 'noise_1/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/noise_strength'.format(r, r)
        else:   # if 'bias_1/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/bias'.format(r, r)
        return o_name

    def handle_const_layer(name):
        if 'const:0' in name:
            o_name = 'G_synthesis_1/4x4/Const/const'
        elif 'conv/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/weight'
        elif 'mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/mod_weight'
        elif 'mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/mod_bias'
        elif 'noise/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/noise_strength'
        else:
            o_name = 'G_synthesis_1/4x4/Conv/bias'
        return o_name

    res, divider = extract_info(w_name)
    if divider == 'ToRGB':
        official_var_name = to_rgb_layer(w_name, res)
    elif divider == 'block':
        official_var_name = handle_block_layer(w_name, res)
    else:   # const
        official_var_name = handle_const_layer(w_name)
    return official_var_name


def variable_name_mapper(g):
    name_mapper = dict()
    for w in g.weights:
        w_name, w_shape = w.name, w.shape

        # mapping layer
        if 'g_mapping' in w_name:
            official_var_name = handle_mapping(w_name)
        elif 'g_synthesis' in w_name:
            official_var_name = handle_synthesis(w_name)
        else:
            official_var_name = 'Gs/dlatent_avg'
            pass    # w_avg

        name_mapper[official_var_name] = w
    return name_mapper


def check_shape(name_mapper, official_vars):
    for official_name, v in name_mapper.items():
        official_shape = [s for n, s in official_vars if n == official_name][0]

        if official_shape == v.shape:
            print('{}: shape matches'.format(official_name))
        else:
            # print(f'Official: {official_name} -> {official_shape}')
            # print(f'Current: {v.name} -> {v.shape}')
            raise ValueError('{}: wrong shape'.format(official_name))
    return


def convert_official_weights():
    g_clone = load_generator(is_g_clone=True)

    # restore official ones to current implementation
    official_checkpoint = tf.train.latest_checkpoint('./official-pretrained')
    official_vars = tf.train.list_variables(official_checkpoint)
    print(official_vars)

    # get name mapper
    name_mapper = variable_name_mapper(g_clone)

    # check shape
    check_shape(name_mapper, official_vars)

    # restore
    tf.compat.v1.train.init_from_checkpoint(official_checkpoint, assignment_map=name_mapper)

    # save
    ckpt_dir = './official-converted'
    ckpt = tf.train.Checkpoint(g_clone=g_clone)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    manager.save(checkpoint_number=0)
    return


def test_generator():
    g_clone = load_generator(is_g_clone=True, ckpt_dir='./official-converted')

    # test
    seed = 6600
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, g_clone.z_dim)
    labels = rnd.randn(1, g_clone.labels_dim)
    latents = latents.astype(np.float32)
    labels = labels.astype(np.float32)
    image_out = g_clone([latents, labels], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out = image_out.numpy()
    Image.fromarray(image_out[0], 'RGB').save('seed{}-restored.png'.format(seed))
    return


def main():
    allow_memory_growth()

    convert_official_weights()
    test_generator()
    return


if __name__ == '__main__':
    main()
