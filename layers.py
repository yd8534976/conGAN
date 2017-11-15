import tensorflow as tf


def conv_bn_relu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                strides=strides, padding='same')
    out_bn = tf.layers.batch_normalization(out_conv)
    out_relu = tf.nn.relu(out_bn)
    return out_relu


def deconv_bn_relu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding='same')
    out_bn = tf.layers.batch_normalization(out_conv)
    out_relu = tf.nn.relu(out_bn)
    return out_relu


def deconv_bn_dropout_relu(inputs, filters, kernel_size=(4, 4), strides=(2, 2)):
    out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding='same')
    out_bn = tf.layers.batch_normalization(out_conv)
    out_dropout = tf.layers.dropout(out_bn)
    out_relu = tf.nn.relu(out_dropout)
    return out_relu