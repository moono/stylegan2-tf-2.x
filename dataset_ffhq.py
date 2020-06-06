import os
import numpy as np
import tensorflow as tf


# n_samples = 70000
def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.dtypes.int64),
        'data': tf.io.FixedLenFeature([], tf.dtypes.string)
    })

    # [0 ~ 255] uint8
    images = tf.io.decode_raw(features['data'], tf.dtypes.uint8)
    images = tf.reshape(images, features['shape'])

    # [0.0 ~ 255.0] float32
    images = tf.cast(images, tf.dtypes.float32)
    return images


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
    res = 64
    batch_size = 4
    epochs = 1
    tfrecord_dir = './tfrecords'

    dataset = get_ffhq_dataset(tfrecord_dir, res, batch_size, epochs)

    for real_images in dataset.take(1):
        print(real_images.shape)
    return


if __name__ == '__main__':
    main()
