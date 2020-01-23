import numpy as np
import tensorflow as tf


def g_logistic_non_saturating(generator, discriminator, z, labels):
    # forward pass
    fake_images = generator([z, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    g_loss = tf.math.softplus(-fake_scores)
    # g_loss = tf.reduce_mean(g_loss)
    return g_loss


def pl_reg(fake_images, w_broadcasted, pl_mean, pl_decay=0.01):
    h, w = fake_images.shape[2], fake_images.shape[3]

    # Compute |J*y|.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch(w_broadcasted)
        pl_noise = tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) / np.sqrt(h * w)

    pl_grads = pl_tape.gradient(tf.reduce_sum(fake_images * pl_noise), w_broadcasted)
    pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)
    return pl_penalty


def d_logistic(generator, discriminator, z, labels, real_images):
    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)
    return d_loss


def r1_reg(discriminator, labels, real_images):
    # simple GP
    with tf.GradientTape() as r1_tape:
        r1_tape.watch(real_images)
        real_loss = tf.reduce_sum(discriminator([real_images, labels], training=True))

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)
    return r1_penalty
