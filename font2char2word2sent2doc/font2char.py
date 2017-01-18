import extenteten as ex
import tensorflow as tf
import tensorflow.contrib.slim as slim


@ex.func_scope()
def font2char(fonts, char_embedding_size, **vgg_hyperparams):
    assert ex.static_rank(fonts) == 3
    return vgg_16(tf.expand_dims(fonts, -1),
                  output_size=char_embedding_size,
                  **vgg_hyperparams)


@ex.func_scope(initializer=tf.contrib.layers.xavier_initializer_conv2d)
def vgg_16(inputs,
           output_size,
           *,
           dropout_keep_prob=0.5,
           mode):
    def dropout(x, scope):
        return slim.dropout(
            x,
            dropout_keep_prob,
            is_training=(mode == tf.contrib.learn.ModeKeys.TRAIN),
            scope=scope)

    def multi_conv(x, num_convs, num_channels, scope):
        return slim.repeat(x,
                           num_convs,
                           slim.conv2d,
                           num_channels,
                           3,
                           scope=scope)

    def pool(x, scope):
        return slim.max_pool2d(x, 2, scope=scope)

    h = multi_conv(inputs, 2, 64, 'conv1')
    h = pool(h, 'pool1')
    h = multi_conv(inputs, 2, 128, 'conv2')
    h = pool(h, 'pool2')
    h = multi_conv(inputs, 3, 256, 'conv3')
    h = pool(h, 'pool3')
    h = multi_conv(inputs, 3, 512, 'conv4')
    h = pool(h, 'pool4')
    h = multi_conv(inputs, 3, 512, 'conv5')
    h = pool(h, 'pool5')

    # Use conv2d instead of fully_connected layers.
    h = slim.conv2d(h, 4096, 7, padding='VALID', scope='fc6')
    h = dropout(h, 'dropout6')
    h = slim.conv2d(h, 4096, 1, scope='fc7')
    h = dropout(h, 'dropout7')
    h = slim.conv2d(h,
                    output_size,
                    [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='fc8')
    h = dropout(h, 'dropout8')

    return tf.reduce_mean(h, [1, 2])
