import tensorflow as tf


def conv_bn_lrelu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                strides=strides, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    out_bn = tf.layers.batch_normalization(out_conv)
    out_lrelu = tf.nn.leaky_relu(out_bn)
    return out_lrelu


def deconv_bn_relu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    out_bn = tf.layers.batch_normalization(out_conv)
    out_relu = tf.nn.relu(out_bn)
    return out_relu


def deconv_bn_dropout_relu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    out_bn = tf.layers.batch_normalization(out_conv)
    out_dropout = tf.layers.dropout(out_bn)
    out_relu = tf.nn.relu(out_dropout)
    return out_relu


def conv(inputs, filters, kernel_size=(4, 4), strides=(1, 1)):
    out_conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                strides=strides, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    out_bn = tf.layers.batch_normalization(out_conv)
    out_lrelu = tf.nn.leaky_relu(out_bn)
    return out_lrelu
