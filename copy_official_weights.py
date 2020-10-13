import os
import tensorflow as tf

from load_models import load_generator, load_discriminator


def handle_mapping(w_name, is_g_clone):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_mapping')
        indicator = splitted[index + 1]
        val = indicator.split('_')[-1]
        return val

    level = extract_info(w_name)
    o_prefix = f'G_mapping_1' if is_g_clone else f'G_mapping'
    if 'w' in w_name:
        official_var_name = f'{o_prefix}/Dense{level}/weight'
    else:
        official_var_name = f'{o_prefix}/Dense{level}/bias'
    return official_var_name


def to_rgb_layer(name, r, o_prefix):
    if 'conv/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/ToRGB/weight'
    elif 'mod_dense/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/ToRGB/mod_weight'
    elif 'mod_bias/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/ToRGB/mod_bias'
    else:   # if 'bias/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/ToRGB/bias'
    return o_name


def handle_block_layer(name, r, o_prefix):
    if 'conv_0/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv0_up/weight'
    elif 'conv_0/mod_dense/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv0_up/mod_weight'
    elif 'conv_0/mod_bias/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv0_up/mod_bias'
    elif 'noise_0/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv0_up/noise_strength'
    elif 'bias_0/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv0_up/bias'
    elif 'conv_1/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv1/weight'
    elif 'conv_1/mod_dense/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv1/mod_weight'
    elif 'conv_1/mod_bias/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv1/mod_bias'
    elif 'noise_1/w:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv1/noise_strength'
    else:   # if 'bias_1/b:0' in name:
        o_name = f'{o_prefix}/{r}x{r}/Conv1/bias'
    return o_name


def handle_const_layer(name, o_prefix):
    if 'const:0' in name:
        o_name = f'{o_prefix}/4x4/Const/const'
    elif 'conv/w:0' in name:
        o_name = f'{o_prefix}/4x4/Conv/weight'
    elif 'mod_dense/w:0' in name:
        o_name = f'{o_prefix}/4x4/Conv/mod_weight'
    elif 'mod_bias/b:0' in name:
        o_name = f'{o_prefix}/4x4/Conv/mod_bias'
    elif 'noise/w:0' in name:
        o_name = f'{o_prefix}/4x4/Conv/noise_strength'
    else:
        o_name = f'{o_prefix}/4x4/Conv/bias'
    return o_name
    
    
def handle_synthesis(w_name, is_g_clone):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_synthesis')
        indicator1 = splitted[index + 1]
        indicator2 = splitted[index + 2]
        r = indicator1.split('x')[1]
        d = indicator2
        return r, d

    res, divider = extract_info(w_name)
    o_prefix = f'G_synthesis_1' if is_g_clone else f'G_synthesis'
    if divider == 'ToRGB':
        official_var_name = to_rgb_layer(w_name, res, o_prefix)
    elif divider == 'block':
        official_var_name = handle_block_layer(w_name, res, o_prefix)
    else:   # const
        official_var_name = handle_const_layer(w_name, o_prefix)
    return official_var_name


def handle_discriminator_layer(w_name):
    def extract_info(name):
        splitted = name.split('/')
        resolution = splitted[1]
        resolution = resolution.split('x')[0]
        return resolution

    res = extract_info(w_name)
    if 'last_dense' in w_name:
        o_name = 'D/Output/weight'
    elif 'last_bias' in w_name:
        o_name = 'D/Output/bias'
    elif 'FromRGB' in w_name:
        o_name = 'D/1024x1024/FromRGB/weight' if 'conv_' in w_name else 'D/1024x1024/FromRGB/bias'
    elif 'skip' in w_name:
        o_name = f'D/{res}x{res}/Skip/weight'
    elif 'dense_' in w_name:
        o_name = f'D/4x4/Dense0/weight'
    elif 'conv_' in w_name:
        if res == '4':
            o_name = f'D/{res}x{res}/Conv/weight'
        else:
            o_name = f'D/{res}x{res}/Conv0/weight' if '_0' in w_name else f'D/{res}x{res}/Conv1_down/weight'
    elif 'bias_' in w_name:
        if res == '4':
            o_name = f'D/{res}x{res}/Conv/bias' if '_0' in w_name else f'D/{res}x{res}/Dense0/bias'
        else:
            o_name = f'D/{res}x{res}/Conv0/bias' if '_0' in w_name else f'D/{res}x{res}/Conv1_down/bias'
    else:
        raise ValueError('Something went wrong!!')
    return o_name


def variable_name_mapper_g(g, is_g_clone):
    name_mapper = dict()
    for w in g.weights:
        w_name, w_shape = w.name, w.shape

        # mapping layer
        if 'g_mapping' in w_name:
            official_var_name = handle_mapping(w_name, is_g_clone)
        elif 'g_synthesis' in w_name:
            official_var_name = handle_synthesis(w_name, is_g_clone)
        else:
            # w_avg
            official_var_name = 'Gs/dlatent_avg' if is_g_clone else 'G/dlatent_avg'

        name_mapper[official_var_name] = w
    return name_mapper


def variable_name_mapper_d(d):
    name_mapper = dict()
    for w in d.weights:
        w_name, w_shape = w.name, w.shape

        official_var_name = handle_discriminator_layer(w_name)
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


def convert_official_generator_weights(ckpt_dir, is_g_clone, use_custom_cuda):
    generator = load_generator(g_params=None, is_g_clone=is_g_clone, ckpt_dir=None, custom_cuda=use_custom_cuda)

    # restore official ones to current implementation
    official_checkpoint = tf.train.latest_checkpoint('./official-pretrained')
    official_vars = tf.train.list_variables(official_checkpoint)

    # get name mapper
    name_mapper = variable_name_mapper_g(generator, is_g_clone=is_g_clone)
    for name_g, tvar in name_mapper.items():
        print(f'{name_g}: {tvar.name}')

    # check shape
    check_shape(name_mapper, official_vars)

    # restore
    tf.compat.v1.train.init_from_checkpoint(official_checkpoint, assignment_map=name_mapper)

    # save
    if is_g_clone:
        ckpt = tf.train.Checkpoint(g_clone=generator)
    else:
        ckpt = tf.train.Checkpoint(generator=generator)
    out_dir = os.path.join(ckpt_dir, 'g_clone' if is_g_clone else 'generator')
    manager = tf.train.CheckpointManager(ckpt, out_dir, max_to_keep=1)
    manager.save(checkpoint_number=0)
    return


def convert_official_discriminator_weights(ckpt_dir, use_custom_cuda):
    discriminator = load_discriminator(d_params=None, ckpt_dir=None, custom_cuda=use_custom_cuda)

    # restore official ones
    official_checkpoint = tf.train.latest_checkpoint('./official-pretrained')
    official_vars = tf.train.list_variables(official_checkpoint)

    # get name mapper
    name_mapper = variable_name_mapper_d(discriminator)
    for name_d, tvar in name_mapper.items():
        print(f'{name_d}: {tvar.name}')

    # check shape
    check_shape(name_mapper, official_vars)

    # restore
    tf.compat.v1.train.init_from_checkpoint(official_checkpoint, assignment_map=name_mapper)

    # save
    ckpt = tf.train.Checkpoint(discriminator=discriminator)
    out_dir = os.path.join(ckpt_dir, 'discriminator')
    manager = tf.train.CheckpointManager(ckpt, out_dir, max_to_keep=1)
    manager.save(checkpoint_number=0)
    return


def convert_official_weights_together(ckpt_dir, use_custom_cuda):
    # instantiate all models
    discriminator = load_discriminator(d_params=None, ckpt_dir=None, custom_cuda=use_custom_cuda)
    generator = load_generator(g_params=None, is_g_clone=False, ckpt_dir=None, custom_cuda=use_custom_cuda)
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=None, custom_cuda=use_custom_cuda)

    # restore official ones
    official_checkpoint = tf.train.latest_checkpoint('./official-pretrained')
    official_vars = tf.train.list_variables(official_checkpoint)
    for name, shape in official_vars:
        print(f'{name}: {shape}')

    # get name mapper
    name_mapper_d = variable_name_mapper_d(discriminator)
    name_mapper_g1 = variable_name_mapper_g(generator, is_g_clone=False)
    name_mapper_g2 = variable_name_mapper_g(g_clone, is_g_clone=True)
    name_mapper = {**name_mapper_d, **name_mapper_g1, **name_mapper_g2}

    # check shape
    check_shape(name_mapper, official_vars)

    # restore
    tf.compat.v1.train.init_from_checkpoint(official_checkpoint, assignment_map=name_mapper)

    # save
    ckpt = tf.train.Checkpoint(discriminator=discriminator,
                               generator=generator,
                               g_clone=g_clone)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    manager.save(checkpoint_number=0)
    return


def main():
    from tf_utils import allow_memory_growth

    allow_memory_growth()

    ckpt_dir_base = './official-converted'
    for use_custom_cuda in [True, False]:
        ckpt_dir = os.path.join(ckpt_dir_base, 'cuda') if use_custom_cuda else os.path.join(ckpt_dir_base, 'ref')

        convert_official_weights_together(ckpt_dir, use_custom_cuda)

        # convert_official_discriminator_weights(ckpt_dir, use_custom_cuda)
        # for is_g_clone in [True, False]:
        #     convert_official_generator_weights(ckpt_dir, is_g_clone, use_custom_cuda)
    return


if __name__ == '__main__':
    main()
