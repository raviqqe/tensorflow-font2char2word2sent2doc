import extenteten as ex
import tensorflow as tf

from .char2word2sent2doc import char2word2sent2doc


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
