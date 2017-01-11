import extenteten as ex
import tensorflow as tf

from . import font2char


@ex.func_scope()
def test_font2char():
    font2char.font2char(tf.zeros([64, 224, 224]),
                        char_embedding_size=100,
                        mode=tf.contrib.learn.ModeKeys.TRAIN)


@ex.func_scope()
def test_vgg_16():
    font2char.vgg_16(tf.zeros([64, 224, 224, 4]),
                     output_size=100,
                     mode=tf.contrib.learn.ModeKeys.TRAIN)
