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
        h1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h1 = tf.layers.batch_normalization(h1)
        h2 = tf.layers.conv2d(h1, filters=128, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h2 = tf.layers.batch_normalization(h2)
        h3 = tf.layers.conv2d(h2, filters=256, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h3 = tf.layers.batch_normalization(h3)
        h4 = tf.layers.conv2d(h3, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h4 = tf.layers.batch_normalization(h4)
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
        h1 = conv_bn_lrelu(inputs, 64)
        h2 = conv_bn_lrelu(h1, 128)
        h3 = conv_bn_lrelu(h2, 256)
        h4 = conv_bn_lrelu(h3, 512)
        logits = tf.layers.dense(h4, units=1)
    return logits


def generator(inputs, name="generator"):
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
        # input 256x256
        inputs = tf.layers.batch_normalization(inputs)
        # h1 128x128
        e1 = conv_bn_lrelu(inputs, 64)
        # h2 64x64
        e2 = conv_bn_lrelu(e1, 128)
        # h3 32x32
        e3 = conv_bn_lrelu(e2, 256)
        # h4 16x16
        e4 = conv_bn_lrelu(e3, 512)
        # h5 8x8
        e5 = conv_bn_lrelu(e4, 512)
        # h6 4x4
        e6 = conv_bn_lrelu(e5, 512)
        # h7 2x2
        e7 = conv_bn_lrelu(e6, 512)
        # h8 1x1
        e8 = conv_bn_lrelu(e7, 512)
        # decoder
        # h9 2x2
        d1 = deconv_bn_dropout_relu(e8, filters=512)
        d1 = tf.concat([d1, e7], axis=3)
        # h10 4x4
        d2 = deconv_bn_dropout_relu(d1, filters=512)
        d2 = tf.concat([d2, e6], axis=3)
        # h11 8x8
        d3 = deconv_bn_dropout_relu(d2, filters=512)
        d3 = tf.concat([d3, e5], axis=3)
        # h12 16x16
        d4 = deconv_bn_relu(d3, filters=512)
        d4 = tf.concat([d4, e4], axis=3)
        # h13 32x32
        d5 = deconv_bn_relu(d4, filters=512)
        d5 = tf.concat([d5, e3], axis=3)
        # h14 64x64
        d6 = deconv_bn_relu(d5, filters=256)
        d6 = tf.concat([d6, e2], axis=3)
        # h15 128x128
        d7 = deconv_bn_relu(d6, filters=128)
        d7 = tf.concat([d7, e1], axis=3)
        # h16 256x256
        d8 = deconv_bn_relu(d7, filters=64)
        # out 256x256x3
        out_rgb = tf.layers.conv2d_transpose(d8, filters=3, kernel_size=(4, 4),
                                             strides=(1, 1), padding='same')
        out = 128 * (tf.nn.tanh(out_rgb) - 1)
    return out
