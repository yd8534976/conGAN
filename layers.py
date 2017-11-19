import tensorflow as tf


# Convolutional-BatchNorm-LeakyReLU
def conv_bn_lrelu(inputs, filters, kernel_size=(5, 5), strides=(2, 2), use_bn=True, name="conv_bn_lrelu"):
    with tf.variable_scope(name):
        out_conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                    strides=strides, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if use_bn:
            out_conv = tf.layers.batch_normalization(out_conv)
        out_lrelu = tf.nn.leaky_relu(out_conv)
    return out_lrelu


# Deconvolutional-BatchNorm-ReLU
def deconv_bn_relu(inputs, filters, kernel_size=(5, 5), strides=(2, 2), name="deconv_bn_relu"):
    with tf.variable_scope(name):
        out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                              strides=strides, padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out_bn = tf.layers.batch_normalization(out_conv)
        out_relu = tf.nn.relu(out_bn)
    return out_relu


# Deconvolutional-BatchNorm-Dropout-ReLU
def deconv_bn_dropout_relu(inputs, filters, kernel_size=(5, 5), strides=(2, 2), name="deconv_bn_dropout_relu"):
    with tf.variable_scope(name):
        out_conv = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size,
                                              strides=strides, padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out_bn = tf.layers.batch_normalization(out_conv)
        out_dropout = tf.layers.dropout(out_bn)
        out_relu = tf.nn.relu(out_dropout)
    return out_relu
