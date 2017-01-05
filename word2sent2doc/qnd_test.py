import sys

import tensorflow as tf

from . import qnd


def test_def_word2sent2doc():
    sys.argv = ["command", "--num_classes", "7", "--word_space_size", "10000"]

    model = qnd.def_word2sent2doc()

    zeros = lambda *shape: tf.zeros(shape, tf.int32)

    document = zeros(12, 34, 56)

    model(document, zeros(12))
    model(document, zeros(12, 10))
