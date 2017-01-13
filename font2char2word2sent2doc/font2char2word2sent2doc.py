import argtyp
import extenteten as ex
import numpy as np
import tensorflow as tf
import qndex

from . import util
from .ar2word2sent2doc import ar2word2sent2doc
from .char2word2sent2doc import add_flags as add_child_flags
from .font2char import font2char


@ex.func_scope()
def font2char2word2sent2doc(document,
                            *,
                            words,
                            fonts,
                            char_embedding_size,
                            dropout_keep_prob,
                            mode,
                            **ar2word2sent2doc_hyperparams):
    assert ex.static_rank(document) == 3
    assert ex.static_rank(words) == 2
    assert ex.static_rank(fonts) == 3

    return ar2word2sent2doc(
        document,
        words=words,
        char_embeddings=font2char(fonts,
                                  char_embedding_size=char_embedding_size,
                                  dropout_keep_prob=dropout_keep_prob,
                                  mode=mode),
        **ar2word2sent2doc_hyperparams)


def add_flags():
    adder = add_child_flags()

    def font_file(filename):
        fonts = np.array(argtyp.json_file(filename), np.float32)
        fonts -= fonts.mean()
        return fonts / np.sqrt((fonts ** 2).mean())

    adder.add_required_flag(
        "font_file",
        dest="fonts",
        type=font_file)
    adder.add_flag("dropout_keep_prob", type=float, default=0.5)

    return adder


def def_font2char2word2sent2doc():
    adder = add_flags()
    classify = qndex.def_classify()
    word_array = util.def_word_array()

    def model(document, label=None, *, mode):
        return classify(
            font2char2word2sent2doc(
                document,
                words=word_array(),
                mode=mode,
                **{key: value for key, value in adder.flags.items()
                   if key not in {"chars", "words"}}),
            label,
            mode=mode)

    return model
