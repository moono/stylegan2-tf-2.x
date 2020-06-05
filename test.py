import tensorflow as tf

from stylegan2.generator import Generator
from stylegan2.discriminator import Discriminator


def main():
    batch_size = 1
    z_dim = 512
    w_dim = 512
    l_dim = 0
    n_mapping = 8

    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]

    g_params = {
        'z_dim': z_dim,
        'w_dim': w_dim,
        'labels_dim': l_dim,
        'n_mapping': n_mapping,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
    }
    d_params = {
        'labels_dim': l_dim,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
    }

    z = tf.random.normal(shape=[batch_size, z_dim])
    l = tf.random.normal(shape=[batch_size, l_dim])
    i = tf.random.normal((batch_size, 3, resolutions[-1], resolutions[-1]))

    generator = Generator(g_params)
    fake_images1, _ = generator([z, l], training=True)
    fake_images2, _ = generator([z, l], training=False)
    generator.summary()

    print(fake_images1.shape)
    print()
    for v in generator.variables:
        print('{}: {}'.format(v.name, v.shape))

    discriminator = Discriminator(d_params)
    scores1 = discriminator([i, l], training=True)
    scores2 = discriminator([i, l], training=False)
    discriminator.summary()

    print(scores1.shape)
    print(scores2.shape)

    print()
    for v in discriminator.variables:
        print('{}: {}'.format(v.name, v.shape))
    return


if __name__ == '__main__':
    main()
