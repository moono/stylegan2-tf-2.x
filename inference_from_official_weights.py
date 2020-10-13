import os
import numpy as np
import tensorflow as tf

from PIL import Image
from stylegan2.utils import postprocess_images
from load_models import load_generator
from copy_official_weights import convert_official_weights_together


def test_generator(ckpt_dir, use_custom_cuda, out_fn):
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)

    # test
    seed = 6600
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, g_clone.z_dim)
    labels = rnd.randn(1, g_clone.labels_dim)
    latents = latents.astype(np.float32)
    labels = labels.astype(np.float32)
    image_out = g_clone([latents, labels], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out = image_out.numpy()

    out_fn = f'seed{seed}-{out_fn}'
    Image.fromarray(image_out[0], 'RGB').save(out_fn)
    return


def main():
    from tf_utils import allow_memory_growth

    allow_memory_growth()

    # common variables
    ckpt_dir_base = './official-converted'

    # saving phase
    for use_custom_cuda in [True, False]:
        ckpt_dir = os.path.join(ckpt_dir_base, 'cuda') if use_custom_cuda else os.path.join(ckpt_dir_base, 'ref')
        convert_official_weights_together(ckpt_dir, use_custom_cuda)

    # inference phase
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
    ckpt_dir_ref = os.path.join(ckpt_dir_base, 'ref')

    # 1. inference cuda saved weight from cuda model
    test_generator(ckpt_dir_cuda, use_custom_cuda=True, out_fn='from-cuda-to-cuda.png')

    # 2. inference cuda saved weight from ref model
    test_generator(ckpt_dir_cuda, use_custom_cuda=False, out_fn='from-cuda-to-ref.png')

    # 3. inference ref saved weight from ref model
    test_generator(ckpt_dir_ref, use_custom_cuda=False, out_fn='from-ref-to-ref.png')

    # 4. inference ref saved weight from cuda model
    test_generator(ckpt_dir_ref, use_custom_cuda=True, out_fn='from-ref-to-cuda.png')
    return


if __name__ == '__main__':
    main()
