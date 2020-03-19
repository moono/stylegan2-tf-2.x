import numpy as np
import tensorflow as tf


def setup_resample_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    return k


def upfirdn_ref(x, k, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    in_height, in_width = tf.shape(x)[1], tf.shape(x)[2]
    minor_dim = tf.shape(x)[3]
    kernel_h, kernel_w = k.shape

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, in_height, 1, in_width, 1, minor_dim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, up_y - 1], [0, 0], [0, up_x - 1], [0, 0]])
    x = tf.reshape(x, [-1, in_height * up_y, in_width * up_x, minor_dim])

    # Pad (crop if negative).
    x = tf.pad(x, [
        [0, 0], 
        [tf.math.maximum(pad_y0, 0), tf.math.maximum(pad_y1, 0)], 
        [tf.math.maximum(pad_x0, 0), tf.math.maximum(pad_x1, 0)], 
        [0, 0]
    ])
    x = x[:, tf.math.maximum(-pad_y0, 0): tf.shape(x)[1] - tf.math.maximum(-pad_y1, 0),
          tf.math.maximum(-pad_x0, 0): tf.shape(x)[2] - tf.math.maximum(-pad_x1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, in_height * up_y + pad_y0 + pad_y1, in_width * up_x + pad_x0 + pad_x1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
    x = tf.reshape(x, [-1,
                       minor_dim,
                       in_height * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                       in_width * up_x + pad_x0 + pad_x1 - kernel_w + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::down_y, ::down_x, :]


def simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0):
    output_channel = tf.shape(x)[1]
    x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3], 1])
    x = upfirdn_ref(x, k,
                    up_x=up, up_y=up, down_x=down, down_y=down, pad_x0=pad0, pad_x1=pad1, pad_y0=pad0, pad_y1=pad1)
    x = tf.reshape(x, [-1, output_channel, tf.shape(x)[1], tf.shape(x)[2]])
    return x


def upsample_conv_2d(x, k, weight, factor, gain):
    x_height, x_width = tf.shape(x)[2], tf.shape(x)[3]
    w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]
    w_ic, w_oc = tf.shape(weight)[2], tf.shape(weight)[3]

    # Setup filter kernel.
    k = k * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (w_width - 1)
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2 + 1

    # Determine data dimensions.
    strides = [1, 1, factor, factor]
    output_shape = [1, w_oc, (x_height - 1) * factor + w_height, (x_width - 1) * factor + w_width]
    num_groups = tf.shape(x)[1] // w_ic

    # Transpose weights.
    weight = tf.reshape(weight, [w_height, w_width, w_ic, num_groups, -1])
    weight = tf.transpose(weight[::-1, ::-1], [0, 1, 4, 3, 2])
    weight = tf.reshape(weight, [w_height, w_width, -1, num_groups * w_ic])

    # Execute.
    x = tf.nn.conv2d_transpose(x, weight, output_shape, strides, padding='VALID', data_format='NCHW')
    x = simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
    return x


def conv_downsample_2d(x, k, weight, factor, gain):
    w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]

    # Setup filter kernel.
    k = k * gain
    p = (k.shape[0] - factor) + (w_width - 1)
    pad0 = (p + 1) // 2
    pad1 = p // 2

    strides = [1, 1, factor, factor]
    x = simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
    x = tf.nn.conv2d(x, weight, strides, padding='VALID', data_format='NCHW')
    return x


def upsample_2d(x, k, factor, gain):
    # Setup filter kernel.
    k = k * (gain * (factor ** 2))
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2

    x = simple_upfirdn_2d(x, k, up=factor, pad0=pad0, pad1=pad1)
    return x


def downsample_2d(x, k, factor, gain):
    # Setup filter kernel.
    k = k * gain
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2
    pad1 = p // 2

    x = simple_upfirdn_2d(x, k, down=factor, pad0=pad0, pad1=pad1)
    return x
