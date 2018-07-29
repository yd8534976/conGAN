from layers import *


def con_discriminator(x, y, is_training=True, name="discriminator"):
    """
    Discriminator for conGAN
    :param x: conditional image
    :param y: target image
    :param name: name for variable scope
    :return: probability if x and y are paired
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        inputs = tf.concat([x, y], axis=3)
        h1 = conv_bn_lrelu(inputs, 64, use_bn=False, name="d_h1_conv")
        h2 = conv_bn_lrelu(h1, 128, name="d_h2_conv", is_training=is_training)
        h3 = conv_bn_lrelu(h2, 256, name="d_h3_conv", is_training=is_training)
        h4 = conv_bn_lrelu(h3, 512, name="d_h4_conv", is_training=is_training)
        logits = tf.layers.dense(h4, units=1, name="d_logits_dense")
    return logits


def generator(inputs, is_training=True, name="generator"):
    """
    Generator for conGAN
    encoder:
        C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
        CD512-CD512-CD512-C512-C512-C256-C128-C64
    :param inputs:
    :param name: name for variable scope
    :return: generated image, scale: (-1, 1)
    """
    with tf.variable_scope(name):
        # inputs 256x256x3
        # encoder
        # convolutional layers

        # e1 128x128x64
        e1 = conv_bn_lrelu(inputs, 64, use_bn=False, name="g_e1_conv", is_training=is_training)

        # e2 64x64x128
        e2 = conv_bn_lrelu(e1, 128, name="g_e2_conv", is_training=is_training)

        # e3 32x32x256
        e3 = conv_bn_lrelu(e2, 256, name="g_e3_conv", is_training=is_training)

        # e4 16x16x512
        e4 = conv_bn_lrelu(e3, 512, name="g_e4_conv", is_training=is_training)

        # e5 8x8x512
        e5 = conv_bn_lrelu(e4, 512, name="g_e5_conv", is_training=is_training)

        # e6 4x4x512
        e6 = conv_bn_lrelu(e5, 512, name="g_e6_conv", is_training=is_training)

        # e7 2x2x512
        e7 = conv_bn_lrelu(e6, 512, name="g_e7_conv", is_training=is_training)

        # e8 1x1x512
        e8 = conv_bn_lrelu(e7, 512, name="g_e8_conv", is_training=is_training)

        # decoder
        # deconvolutional layers

        # d1 2x2x512*2
        d1 = deconv_bn_dropout_relu(e8, filters=512, name="g_d1_deconv", is_training=is_training)
        d1 = tf.concat([d1, e7], axis=3)

        # d2 4x4x512*2
        d2 = deconv_bn_dropout_relu(d1, filters=512, name="g_d2_deconv", is_training=is_training)
        d2 = tf.concat([d2, e6], axis=3)

        # d3 8x8x512*2
        d3 = deconv_bn_dropout_relu(d2, filters=512, name="g_d3_deconv", is_training=is_training)
        d3 = tf.concat([d3, e5], axis=3)

        # d4 16x16x512*2
        d4 = deconv_bn_relu(d3, filters=512, name="g_d4_deconv", is_training=is_training)
        d4 = tf.concat([d4, e4], axis=3)

        # d5 32x32x512*2
        d5 = deconv_bn_relu(d4, filters=512, name="g_d5_deconv", is_training=is_training)
        d5 = tf.concat([d5, e3], axis=3)

        # d6 64x64x256*2
        d6 = deconv_bn_relu(d5, filters=256, name="g_d6_deconv", is_training=is_training)
        d6 = tf.concat([d6, e2], axis=3)

        # d7 128x128x128*2
        d7 = deconv_bn_relu(d6, filters=128, name="g_d7_deconv", is_training=is_training)
        d7 = tf.concat([d7, e1], axis=3)

        # d8 256x256x64*2
        d8 = deconv_bn_relu(d7, filters=64, name="g_d8_deconv", is_training=is_training)

        # conv map features to RGB channels
        out = tf.layers.conv2d(d8, filters=3, kernel_size=(5, 5), strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.tanh(out)
    return out
