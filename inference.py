import tensorflow as tf

from PIL import Image
from load_models import load_generator
from tf_utils import allow_memory_growth


def inference(ckpt_dir, use_custom_cuda, res, out_fn=None):
    # create generator
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    filter_index = resolutions.index(res)
    g_params = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': resolutions[:filter_index + 1],
        'featuremaps': featuremaps[:filter_index + 1],
    }
    generator = load_generator(g_params, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)

    # generate image
    fake_images = generator([tf.random.normal(shape=[1, g_params['z_dim']]),
                             tf.random.normal(shape=[1, g_params['labels_dim']])],
                            training=False, truncation_psi=0.5)
    fake_images = (tf.clip_by_value(fake_images, -1.0, 1.0) + 1.0) * 127.5
    fake_images = tf.transpose(fake_images, perm=[0, 2, 3, 1])
    fake_images = tf.cast(fake_images, tf.uint8)
    fake_image = fake_images[0].numpy()

    image = Image.fromarray(fake_image)
    image = image.convert('RGB')
    image.show()
    if out_fn is not None:
        image.save(out_fn)
    return


def main():
    allow_memory_growth()

    checkpoints = [
        {
            'res': 1024,
            'ckpt_dir': './official-converted/cuda',
            'use_custom_cuda': True,
            'out_fn': None,
        },
        {
            'res': 256,
            'ckpt_dir': '/mnt/vision-nas/moono/trained_models/stylegan2-tf-2.x/gold/stylegan2-ffhq-256x256',
            'use_custom_cuda': True,
            'out_fn': 'out_256x256_0.png',
        },
    ]

    for run_item in checkpoints:
        res = run_item['res']
        ckpt_dir = run_item['ckpt_dir']
        use_custom_cuda = run_item['use_custom_cuda']
        out_fn = run_item['out_fn']
        message = f'{res}x{res} with custom cuda' if use_custom_cuda else f'{res}x{res} without custom cuda'

        print(message)
        inference(ckpt_dir, use_custom_cuda, res, out_fn)
    return


if __name__ == '__main__':
    main()
