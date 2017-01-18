import sys

import tensorflow as tf

from .font2char2word2sent2doc import def_font2char2word2sent2doc


def test_def_font2char2word2sent2doc():
    sys.argv = ["command",
                "--num_classes", "7",
                "--font_file", "data/font.ttf",
                "--char_file", "data/chars.txt",
                "--word_file", "data/words.txt"]

    model = def_font2char2word2sent2doc()

    zeros = lambda *shape: tf.zeros(shape, tf.int32)

    document = zeros(12, 34, 56)

    with tf.variable_scope("model0"):
        model(document, zeros(12), mode=tf.contrib.learn.ModeKeys.TRAIN)

    with tf.variable_scope("model1"):
        model(document, zeros(12, 10), mode=tf.contrib.learn.ModeKeys.TRAIN)
