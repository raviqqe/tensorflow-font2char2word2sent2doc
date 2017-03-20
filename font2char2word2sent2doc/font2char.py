import extenteten as ex
import tensorflow as tf


@ex.func_scope()
def font2char(fonts):
    assert ex.static_rank(fonts) == 3

    h = tf.expand_dims(fonts, -1)
    for index, num_channels in enumerate([32, 32, 32]):
        h = tf.contrib.slim.conv2d(
            h, num_channels, 3, scope='conv{}'.format(index))
        h = tf.contrib.slim.max_pool2d(h, 2, 2, scope='pool{}'.format(index))
    h = tf.contrib.slim.flatten(h)

    ex.summary.image(tf.expand_dims(tf.expand_dims(h[:256], 0), 3))

    return h
