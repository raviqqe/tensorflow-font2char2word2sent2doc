import extenteten as ex
import tensorflow as tf

from . import font2char


@ex.func_scope()
def test_font2char():
    assert (
        ex.static_rank(font2char.font2char(
            tf.zeros([64, 224, 224]),
            nums_of_channels=[32] * 4,
            nums_of_attention_channels=[32] * 3))
        == 2)


@ex.func_scope()
def test_attend_to_image():
    assert (
        ex.static_rank(font2char._attend_to_image(
            tf.zeros([64, 224, 224, 1]),
            nums_of_channels=[32] * 3))
        == 4)
