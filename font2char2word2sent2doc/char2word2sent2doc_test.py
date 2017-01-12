import sys

import extenteten as ex
import tensorflow as tf

from .char2word2sent2doc import char2word2sent2doc, def_char2word2sent2doc


def test_char2word2sent2doc():
    char2word2sent2doc(tf.zeros([64, 40, 20], tf.int32),
                       words=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]],
                                         tf.int32),
                       char_space_size=2,
                       char_embedding_size=100,
                       word_embedding_size=100,
                       sentence_embedding_size=100,
                       document_embedding_size=100,
                       context_vector_size=100)


def test_def_char2word2sent2doc():
    sys.argv = [
        "command",
        "--num_classes", "7",
        "--char_file", "data/chars.txt",
        "--word_file", "data/words.txt",
    ]

    model = def_char2word2sent2doc()

    zeros = lambda *shape: tf.zeros(shape, tf.int32)

    document = zeros(12, 34, 56)

    with tf.variable_scope("model0"):
        model(document, zeros(12), mode=tf.contrib.learn.ModeKeys.TRAIN)

    with tf.variable_scope("model1"):
        model(document, zeros(12, 10), mode=tf.contrib.learn.ModeKeys.TRAIN)
