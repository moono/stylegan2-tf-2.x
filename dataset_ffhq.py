import os
import numpy as np
import tensorflow as tf


# n_samples = 70000
def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.dtypes.int64),
        'data': tf.io.FixedLenFeature([], tf.dtypes.string)
    })

    # [0 ~ 255] uint8 -> [-1.0 ~ 1.0] float32
    image = tf.io.decode_raw(features['data'], tf.dtypes.uint8)
    image = tf.reshape(image, features['shape'])
    image = tf.transpose(image, perm=[1, 2, 0])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.dtypes.float32)
    image = image / 127.5 - 1.0
    image = tf.transpose(image, perm=[2, 0, 1])
    return image


def get_ffhq_dataset(tfrecord_base_dir, res, batch_size, epochs=None, buffer_size=1000):
    fn_index = int(np.log2(res))
    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

    # with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(map_func=parse_tfrecord_tf, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    from PIL import Image

    res = 64
    batch_size = 4
    epochs = 1
    tfrecord_dir = './tfrecords'

    dataset = get_ffhq_dataset(tfrecord_dir, res, batch_size, epochs)

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
