from layers import *


# discriminator for cGAN
# 70 x 70
def discriminator(inputs, name="discriminator"):
    """
    Discriminator for conGAN
    :param input:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        #inputs = tf.random_crop(inputs, [1, 70, 70, 6])
        h1 = conv_bn_lrelu(inputs, 64, use_bn=False)
        h2 = conv_bn_lrelu(h1, 128)
        h3 = conv_bn_lrelu(h2, 256)
        h4 = conv_bn_lrelu(h3, 512)
        logits = tf.layers.dense(h4, units=1)
    return logits


def con_discriminator(x, y, name="discriminator"):
    """
    Discriminator for conGAN
    :param input:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        inputs = tf.concat([x, y], axis=3)
        #inputs = tf.random_crop(inputs, [1, 70, 70, 6])
        h1 = conv_bn_lrelu(inputs, 64, use_bn=False)
        h2 = conv_bn_lrelu(h1, 128)
        h3 = conv_bn_lrelu(h2, 256)
        h4 = conv_bn_lrelu(h3, 512)
        logits = tf.layers.dense(h4, units=1)
    return logits


def generator(x, z, name="generator"):
    """
    Generator for conGAN
    encoder:
        C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
        CD512-CD512-CD512-C512-C512-C256-C128-C64
    :param input:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        # encoder
        # input 256x256x3
        inputs = tf.concat([x, z], axis=3)

        # e1 128x128x64
        e1 = conv_bn_lrelu(inputs, 64, use_bn=False)

        # e2 64x64x128
        e2 = conv_bn_lrelu(e1, 128)

        # e3 32x32x256
        e3 = conv_bn_lrelu(e2, 256)

        # e4 16x16x512
        e4 = conv_bn_lrelu(e3, 512)

        # e5 8x8x512
        e5 = conv_bn_lrelu(e4, 512)

        # e6 4x4x512
        e6 = conv_bn_lrelu(e5, 512)

        # e7 2x2x512
        e7 = conv_bn_lrelu(e6, 512)

        # e8 1x1x512
        e8 = conv_bn_lrelu(e7, 512)

        # decoder

        # d1 2x2x512*2
        d1 = deconv_bn_dropout_relu(e8, filters=512)
        d1 = tf.concat([d1, e7], axis=3)

        # d2 4x4x512*2
        d2 = deconv_bn_dropout_relu(d1, filters=512)
        d2 = tf.concat([d2, e6], axis=3)

        # d3 8x8x512*2
        d3 = deconv_bn_dropout_relu(d2, filters=512)
        d3 = tf.concat([d3, e5], axis=3)

        # d4 16x16x512*2
        d4 = deconv_bn_relu(d3, filters=512)
        d4 = tf.concat([d4, e4], axis=3)

        # d5 32x32x512*2
        d5 = deconv_bn_relu(d4, filters=512)
        d5 = tf.concat([d5, e3], axis=3)

        # d6 64x64x256*2
        d6 = deconv_bn_relu(d5, filters=256)
        d6 = tf.concat([d6, e2], axis=3)

        # d7 128x128x128*2
        d7 = deconv_bn_relu(d6, filters=128)
        d7 = tf.concat([d7, e1], axis=3)

        # d8 256x256x64*2
        d8 = deconv_bn_relu(d7, filters=64)
        #d8 = tf.concat([d8, inputs], axis=3)
        # out 256x256x3
        dd1 = deconv_bn_relu(e2, filters=128)
        dd2 = deconv_bn_relu(dd1, filters=64)
        out = tf.layers.conv2d(dd2, filters=3, kernel_size=(3, 3), strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.tanh(out)
    return out
