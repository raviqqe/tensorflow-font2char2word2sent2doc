import extenteten as ex
import tensorflow as tf

from . import font2char


@ex.func_scope()
def test_font2char():
    font2char.font2char(tf.zeros([64, 224, 224]))
