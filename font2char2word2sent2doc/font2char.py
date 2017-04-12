import extenteten as ex
import tensorflow as tf


@ex.func_scope()
def font2char(font, num_layers=4, num_channels=32):
    assert ex.static_rank(font) == 3

    h = _attend_to_image(tf.expand_dims(font, -1), num_layers=num_layers - 1)

    for index, num_channels in enumerate([num_channels] * num_layers):
        h = tf.contrib.slim.conv2d(
            h, num_channels, 3, scope='conv{}'.format(index))
        h = tf.contrib.slim.max_pool2d(h, 2, 2, scope='pool{}'.format(index))
    h = tf.contrib.slim.flatten(h)

    ex.summary.image(tf.expand_dims(tf.expand_dims(h[:256], 0), 3))

    return h


@ex.func_scope()
def _attend_to_image(images, num_layers=3, num_channels=32):
    assert ex.static_rank(images) == 4

    return tf.transpose(
        tf.transpose(images) *
        tf.transpose(_calculate_attention(images,
                                          num_layers=num_layers,
                                          num_channels=num_channels)))


@ex.func_scope()
def _calculate_attention(images, num_layers=3, num_channels=32):
    assert ex.static_rank(images) == 4

    h = images

    for index, num_channels in enumerate([num_channels] * num_layers):
        h = tf.contrib.slim.conv2d(
            h, num_channels, 3, scope='conv{}'.format(index))
        h = tf.contrib.slim.max_pool2d(h, 2, 2, scope='pool{}'.format(index))

    logits = tf.image.resize_nearest_neighbor(
        tf.expand_dims(tf.reduce_sum(h, axis=3), axis=-1),
        tf.shape(images)[1:3])

    return tf.reshape(tf.nn.softmax(tf.reshape(logits,
                                               [tf.shape(logits)[0], -1])),
                      tf.shape(logits))
