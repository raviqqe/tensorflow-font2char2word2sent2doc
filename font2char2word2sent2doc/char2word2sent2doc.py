import argtyp
import extenteten as ex
import numpy as np
import qnd
import qndex
import tensorflow as tf

from .word2sent2doc import word2sent2doc, add_flags as add_child_flags
from .ar2word2sent2doc import ar2word2sent2doc


UNKNOWN_CHAR_INDEX = 1


@ex.func_scope()
def char2word2sent2doc(document,
                       *,
                       words,
                       char_space_size,
                       char_embedding_size,
                       **ar2word2sent2doc_hyperparams):
    """
    The argument `document` is in the shape of
    (#examples, #sentences per document, #words per sentence).
    """

    assert ex.static_rank(document) == 3
    assert ex.static_rank(words) == 2

    with tf.variable_scope("char_embeddings"):
        char_embeddings = ex.embeddings(id_space_size=char_space_size,
                                        embedding_size=char_embedding_size,
                                        name="char_embeddings")

    return ar2word2sent2doc(document,
                            words=words,
                            char_embeddings=char_embeddings,
                            **ar2word2sent2doc_hyperparams)


def add_flags():
    adder = add_child_flags()

    adder.add_required_flag("char_file", dest="chars", type=argtyp.file_lines)
    adder.add_flag("char_embedding_size", type=int, default=100)

    return adder


def def_char2word2sent2doc():
    adder = add_flags()
    classify = qndex.def_classify()

    word_array = np.zeros([len(qnd.FLAGS.words),
                           max(len(word) for word in qnd.FLAGS.words)],
                          np.int32)

    for i, word in enumerate(qnd.FLAGS.words):
        for j, char in enumerate(word):
            word_array[i, j] = (qnd.FLAGS.chars.index(char)
                                if char in qnd.FLAGS.chars else
                                UNKNOWN_CHAR_INDEX)

    def model(document, label):
        return classify(
            char2word2sent2doc(
                document,
                words=word_array,
                char_space_size=len(qnd.FLAGS.chars),
                **{key: value for key, value in adder.flags.items()
                   if key not in {"chars", "words"}}),
            label)

    return model
