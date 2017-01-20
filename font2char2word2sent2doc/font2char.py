import extenteten as ex
import tensorflow as tf


@ex.func_scope()
def font2char(fonts, char_embedding_size):
    assert ex.static_rank(fonts) == 3
    h = ex.lenet(tf.expand_dims(fonts, -1), output_size=char_embedding_size)
    ex.summary.image(tf.expand_dims(tf.expand_dims(h[:256], 0), 3))
    return h
