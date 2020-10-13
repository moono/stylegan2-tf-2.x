import tensorflow as tf


def d_logistic(real_images, generator, discriminator, z_dim, labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)
    return d_loss


def d_logistic_r1_reg(real_images, generator, discriminator, z_dim, labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)

    # gradient penalty
    with tf.GradientTape() as r1_tape:
        r1_tape.watch([real_images, labels])
        real_loss = tf.reduce_sum(discriminator([real_images, labels], training=True))

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)
    return d_loss, r1_penalty


def g_logistic_non_saturating(real_images, generator, discriminator, z_dim, labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    g_loss = tf.math.softplus(-fake_scores)
    return g_loss


def g_logistic_ns_pathreg(real_images, generator, discriminator, z_dim,
                          pl_mean, pl_minibatch_shrink, pl_denorm, pl_decay,
                          labels=None):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    if labels is None:
        labels = tf.random.normal(shape=[batch_size, 0], dtype=tf.float32)

    pl_minibatch = tf.maximum(1, tf.math.floordiv(batch_size, pl_minibatch_shrink))
    pl_z = tf.random.normal(shape=[pl_minibatch, z_dim], dtype=tf.float32)
    if labels is None:
        pl_labels = tf.random.normal(shape=[pl_minibatch, 0], dtype=tf.float32)
    else:
        pl_labels = labels[:pl_minibatch]

    # forward pass
    fake_images, w_broadcasted = generator([z, labels], ret_w_broadcasted=True, training=True)
    fake_scores = discriminator([fake_images, labels], training=True)
    g_loss = tf.math.softplus(-fake_scores)

    # Evaluate the regularization term using a smaller minibatch to conserve memory.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch([pl_z, pl_labels])
        pl_fake_images, pl_w_broadcasted = generator([pl_z, pl_labels], ret_w_broadcasted=True, training=True)

        pl_noise = tf.random.normal(tf.shape(pl_fake_images)) * pl_denorm
        pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

    pl_grads = pl_tape.gradient(pl_noise_applied, pl_w_broadcasted)
    pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)
    return g_loss, pl_penalty
