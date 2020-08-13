import tensorflow as tf


def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = tf.reduce_prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef


def lerp(a, b, t):
    out = a + (b - a) * t
    return out


def lerp_clip(a, b, t):
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out
