import argtyp
import extenteten as ex
import numpy as np
import qnd
import qndex
import tensorflow as tf

from .word2sent2doc import add_flags as add_child_flags
from .ar2word2sent2doc import ar2word2sent2doc


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
    adder.add_flag("char_embedding_size", type=int, default=100)
    return adder


def def_char2word2sent2doc():
    adder = add_flags()
    classify = qndex.def_classify()
    word_array = qndex.nlp.def_word_array()

    def model(document, label=None, *, mode, key=None):
        return classify(
            char2word2sent2doc(
                document,
                words=word_array(),
                char_space_size=len(qnd.FLAGS.chars),
                **adder.flags),
            label,
            key=key,
            mode=mode)

    return model
