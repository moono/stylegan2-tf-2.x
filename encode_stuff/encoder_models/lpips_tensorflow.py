# import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dropout, Conv2D, Permute
from tensorflow.keras.applications.vgg16 import VGG16


def image_preprocess(image):
    factor = 255.0 / 2.0
    center = 1.0
    scale = tf.constant([0.458, 0.448, 0.450])[None, None, None, :]
    shift = tf.constant([-0.030, -0.088, -0.188])[None, None, None, :]

    image = image / factor - center  # [0.0 ~ 255.0] -> [-1.0 ~ 1.0]
    image = (image - shift) / scale
    return image


def learned_perceptual_metric_model(image_size, vgg_model_ckpt_fn, lin_model_ckpt_fn):
    # initialize all models
    net = perceptual_model(image_size)
    lin = linear_model(image_size)
    net.load_weights(vgg_model_ckpt_fn)
    lin.load_weights(lin_model_ckpt_fn)

    # merge two model
    input1 = Input(shape=(image_size, image_size, 3), dtype='float32', name='input1')
    input2 = Input(shape=(image_size, image_size, 3), dtype='float32', name='input2')

    # preprocess input images
    net_out1 = Lambda(lambda x: image_preprocess(x))(input1)
    net_out2 = Lambda(lambda x: image_preprocess(x))(input2)

    # run vgg model first
    net_out1 = net(net_out1)
    net_out2 = net(net_out2)

    # nhwc -> nchw
    net_out1 = [Permute(dims=(3, 1, 2))(t) for t in net_out1]
    net_out2 = [Permute(dims=(3, 1, 2))(t) for t in net_out2]

    # normalize
    net_out1 = [Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True)))(t)
                for t in net_out1]
    net_out2 = [Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True)))(t)
                for t in net_out2]

    # subtract
    diffs = [Lambda(lambda x: tf.square(x[0] - x[1]))([t1, t2]) for t1, t2 in zip(net_out1, net_out2)]

    # run on learned linear model
    lin_out = lin(diffs)

    # take spatial average
    lin_out = [Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3], keepdims=True))(t) for t in lin_out]

    # take sum of all layers
    lin_out = Lambda(lambda x: tf.reduce_sum(x))(lin_out)

    final_model = Model(inputs=[input1, input2], outputs=lin_out)
    return final_model


# [64, 64, 3] -> [64, 64, 3, 1] -> [1, 3, 64, 64]
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
# )
#
# Sequential(
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
# )
#
# Sequential(
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
# )
#
# Sequential(
#   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace=True)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True)
# )
#
# Sequential(
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace=True)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace=True)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)
# )


# tf.keras.applications
def perceptual_model(image_size):
    # (None, 64, 64, 64)
    # (None, 32, 32, 128)
    # (None, 16, 16, 256)
    # (None, 8, 8, 512)
    # (None, 4, 4, 512)
    layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))

    vgg16_output_layers = [l.output for l in vgg16.layers if l.name in layers]
    model = Model(vgg16.input, vgg16_output_layers, name='perceptual_model')
    return model


# NetLinLayer(
#   (model): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   )
# )
#
# NetLinLayer(
#   (model): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   )
# )
#
# NetLinLayer(
#   (model): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   )
# )
#
# NetLinLayer(
#   (model): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   )
# )
#
# NetLinLayer(
#   (model): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   )
# )

# functional api
def linear_model(input_image_size):
    assert isinstance(input_image_size, int)

    vgg_channels = [64, 128, 256, 512, 512]
    inputs, outputs = list(), list()
    for ii, channel in enumerate(vgg_channels):
        name = 'lin{}'.format(ii)
        image_size = input_image_size // (2 ** ii)

        model_input = Input(shape=(channel, image_size, image_size), dtype='float32')
        model_output = Dropout(rate=0.5, dtype='float32')(model_input)
        model_output = Conv2D(filters=1, kernel_size=1, strides=1, use_bias=False, dtype='float32',
                              data_format='channels_first', name=name)(model_output)
        inputs.append(model_input)
        outputs.append(model_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear_model')
    return model
