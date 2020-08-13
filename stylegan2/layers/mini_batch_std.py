import tensorflow as tf


class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, inputs, training=None, mask=None):
        s = tf.shape(inputs)
        group_size = tf.minimum(self.group_size, s[0])

        y = tf.reshape(inputs, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, inputs.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])

        x = tf.concat([inputs, y], axis=1)
        return x

    def get_config(self):
        config = super(MinibatchStd, self).get_config()
        config.update({
            'group_size': self.group_size,
            'num_new_features': self.num_new_features,
        })
        return config
