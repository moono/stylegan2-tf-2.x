import os
import glob
import numpy as np
import tensorflow as tf


def parse_data(image_fn, res):
    # [0 ~ 255] uint8 -> [-1.0 ~ 1.0] float32
    image = tf.io.read_file(image_fn)
    image = tf.image.decode_png(image, channels=3)
    image = tf.reshape(image, [500, 500, 3])
    image = tf.image.resize(image, [res, res])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.dtypes.float32)
    image = image / 127.5 - 1.0
    image = tf.transpose(image, perm=[2, 0, 1])
    return image


def get_cartoonset_dataset(image_dir, res, batch_size, epochs=None, buffer_size=1000):
    image_fns = glob.glob(os.path.join(image_dir, '*.png'))
    image_fns = sorted(image_fns)

    # with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices((image_fns,))
    dataset = dataset.map(map_func=lambda fn: parse_data(fn, res), num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    from PIL import Image

    # https://google.github.io/cartoonset/index.html
    res = 1024
    batch_size = 4
    epochs = 1
    cartoon_set_images_dir = '/home/mookyung/Downloads/cartoonset10k'

    dataset = get_cartoonset_dataset(cartoon_set_images_dir, res, batch_size, epochs)

    for real_images in dataset.take(4):
        # real_images: [batch_size, 3, res, res] (-1.0 ~ 1.0) float32
        print(real_images.shape)

        images = real_images.numpy()
        images = np.transpose(images, axes=(0, 2, 3, 1))
        images = (images + 1.0) * 127.5
        images = images.astype(np.uint8)
        image = Image.fromarray(images[0])
        image.show()
    return


if __name__ == '__main__':
    main()
