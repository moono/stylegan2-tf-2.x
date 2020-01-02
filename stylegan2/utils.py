import numpy as np
import tensorflow as tf


def print_model_details(model):
    model.summary()

    print('Trainable variables: {}'.format(model.name))
    for v in model.trainable_variables:
        print('\t{}: {}'.format(v.name, v.shape))

    print('None-trainable variables: {}'.format(model.name))
    for v in model.non_trainable_variables:
        print('\t{}: {}'.format(v.name, v.shape))
    return


def check_weight_name_and_shape(prev, curr):
    def filter_name(name):
        name_components = name.split('/')
        name_components = [s.split('_')[0] for s in name_components]
        return '/'.join(name_components)

    print('{} -> {}'.format(prev.name, curr.name))
    for prev_w, curr_w in zip(prev.weights, curr.weights):
        assert filter_name(prev_w.name) == filter_name(curr_w.name)
        assert prev_w.shape == curr_w.shape
        print('{} {} -> {} {}'.format(prev_w.name, prev_w.shape, curr_w.name, curr_w.shape))
    print()
    return


def compute_training_image_counts(train_start_res, resolutions, do_transition_res,
                                  train_trans_images_per_res, train_fixed_images_per_res, train_total_n_images):
    cur_image_count = 0
    train_n_images = dict()
    for res, do_transition in do_transition_res.items():
        if res < train_start_res:
            n_trans = 0
            n_fixed = 0
        else:
            if res != resolutions[-1]:
                n_trans = train_trans_images_per_res if do_transition else 0
                n_fixed = train_fixed_images_per_res
            else:
                n_trans = train_trans_images_per_res if do_transition else 0
                n_fixed = train_total_n_images - (cur_image_count + n_trans)

        train_n_images[res] = {
            'trans': n_trans,
            'fixed': n_fixed,
            # 'total': n_trans + n_fixed
        }

        # update
        cur_image_count = cur_image_count + train_n_images[res]['trans'] + train_n_images[res]['fixed']

    double_check = [v['trans'] + v['fixed'] for k, v in train_n_images.items()]
    assert sum(double_check) == train_total_n_images
    return train_n_images


def lerp(a, b, t):
    out = a + (b - a) * t
    return out


def lerp_clip(a, b, t):
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def random_flip_left_right_nchw(images):
    s = tf.shape(images)
    mask = tf.random.uniform([s[0], 1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [1, s[1], s[2], s[3]])
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[3]))
    return images


def preprocess_fit_train_image(images, res):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    images = random_flip_left_right_nchw(images)
    images.set_shape([None, 3, res, res])
    return images


def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images


def merge_batch_images(images, res, rows, cols):
    batch_size = images.shape[0]
    assert rows * cols == batch_size
    canvas = np.zeros(shape=[res * rows, res * cols, 3], dtype=np.uint8)
    for row in range(rows):
        y_start = row * res
        for col in range(cols):
            x_start = col * res
            index = col + row * cols
            canvas[y_start:y_start + res, x_start:x_start + res, :] = images[index, :, :, :]
    return canvas


def main():
    # out_dtype = tf.dtypes.float32
    # range_in = (0.0, 255.0)
    # range_out = (-1.0, 1.0)
    # # images = np.ones((2, 2), dtype=np.float32) * 255.0
    # images1 = np.ones((2, 2), dtype=np.float32) * 0.0
    # images2 = np.ones((2, 2), dtype=np.float32) * 127.5
    # images3 = np.ones((2, 2), dtype=np.float32) * 255.0
    # adjusted1 = adjust_dynamic_range(images1, range_in, range_out, out_dtype)
    # adjusted2 = adjust_dynamic_range(images2, range_in, range_out, out_dtype)
    # adjusted3 = adjust_dynamic_range(images3, range_in, range_out, out_dtype)
    # print(adjusted1)
    # print(adjusted2)
    # print(adjusted3)
    #
    # out_dtype = tf.dtypes.uint8
    # range_in = (-1.0, 1.0)
    # range_out = (0.0, 255.0)
    # images1 = np.ones((2, 2), dtype=np.float32) * -1.0
    # images2 = np.ones((2, 2), dtype=np.float32) * 0.0
    # images3 = np.ones((2, 2), dtype=np.float32) * 1.0
    # adjusted1 = adjust_dynamic_range(images1, range_in, range_out, out_dtype)
    # adjusted2 = adjust_dynamic_range(images2, range_in, range_out, out_dtype)
    # adjusted3 = adjust_dynamic_range(images3, range_in, range_out, out_dtype)
    # print(adjusted1)
    # print(adjusted2)
    # print(adjusted3)

    batch_size = 8
    res = 128
    fake_images = np.ones(shape=(batch_size, res, res, 3), dtype=np.uint8)
    merge_batch_images(fake_images, res, rows=4, cols=2)
    return


if __name__ == '__main__':
    main()
