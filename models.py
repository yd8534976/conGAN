import tensorflow as tf


# discriminator for cGAN
# 70 x 70
def discriminator(input, name="discriminator"):
    """
    Discriminator for conGAN
    :param input:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        h1 = tf.layers.conv2d(input, filters=64, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1, filters=128, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2, filters=256, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.relu)
        h4 = tf.layers.conv2d(h3, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.relu)
        logits = tf.layers.dense(h4, units=1, activation=tf.nn.sigmoid)

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
        inputs = tf.layers.batch_normalization(inputs)
        h1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        inputs = tf.layers.batch_normalization(inputs)
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
        h5 = tf.layers.conv2d(h4, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h5 = tf.layers.batch_normalization(h5)
        h6 = tf.layers.conv2d(h5, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h6 = tf.layers.batch_normalization(h6)
        h7 = tf.layers.conv2d(h6, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h7 = tf.layers.batch_normalization(h7)
        h8 = tf.layers.conv2d(h7, filters=512, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              activation=tf.nn.leaky_relu)
        h8 = tf.layers.batch_normalization(h8)
        # decoder
        h9 = tf.layers.conv2d_transpose(h8, filters=512, kernel_size=(4, 4),
                                        strides=(2, 2), padding='same',
                                        activation=tf.nn.relu)
        h9 = tf.concat([h7, h9], axis=3)
        h9 = tf.layers.batch_normalization(h9)
        h9 = tf.nn.dropout(h9, keep_prob=0.5)
        h10 = tf.layers.conv2d_transpose(h9, filters=512, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h10 = tf.concat([h6, h10], axis=3)
        h10 = tf.layers.batch_normalization(h10)
        h10 = tf.nn.dropout(h10, keep_prob=0.5)
        h11 = tf.layers.conv2d_transpose(h10, filters=512, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h11 = tf.concat([h5, h11], axis=3)
        h11 = tf.layers.batch_normalization(h11)
        h11 = tf.nn.dropout(h11, keep_prob=0.5)
        h12 = tf.layers.conv2d_transpose(h11, filters=512, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h12 = tf.concat([h4, h12], axis=3)
        h12 = tf.layers.batch_normalization(h12)
        h13 = tf.layers.conv2d_transpose(h12, filters=512, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h13 = tf.concat([h3, h13], axis=3)
        h13 = tf.layers.batch_normalization(h13)
        h14 = tf.layers.conv2d_transpose(h13, filters=256, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h14 = tf.concat([h2, h14], axis=3)
        h14 = tf.layers.batch_normalization(h14)
        h15 = tf.layers.conv2d_transpose(h14, filters=128, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)
        h15 = tf.concat([h1, h15], axis=3)
        h15 = tf.layers.batch_normalization(h15)
        h16 = tf.layers.conv2d_transpose(h15, filters=64, kernel_size=(4, 4),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)

        out = tf.layers.conv2d(h16, filters=3, kernel_size=(4, 4),
                               strides=(1, 1), padding='same',
                               activation=tf.nn.relu)

    return out
