import json

import char2image
import extenteten as ex
import numpy as np
import tensorflow as tf
import qnd
import qndex

from .ar2word2sent2doc import ar2word2sent2doc
from .char2word2sent2doc import add_flags as add_child_flags
from .font2char import font2char


@ex.func_scope()
def font2char2word2sent2doc(document,
                            *,
                            words,
                            fonts,
                            dropout_keep_prob,
                            mode,
                            **ar2word2sent2doc_hyperparams):
    assert ex.static_rank(document) == 3
    assert ex.static_rank(words) == 2
    assert ex.static_rank(fonts) == 3

    return ar2word2sent2doc(
        document,
        words=words,
        char_embeddings=font2char(fonts),
        **ar2word2sent2doc_hyperparams)


def add_flags():
    adder = add_child_flags()
    adder.add_flag("dropout_keep_prob", type=float, default=0.5)

    qnd.add_required_flag("font_file")
    qnd.add_flag("font_size", type=int, default=32)
    qnd.add_flag("save_font_array_file")
    qnd.add_flag("regularization_scale", type=float, default=1e-8)

    return adder


def font_array():
    fonts = np.array(
        char2image.chars_to_images(
            qnd.FLAGS.chars,
            char2image.filename_to_font(qnd.FLAGS.font_file,
                                        qnd.FLAGS.font_size)),
        np.float32)

    if qnd.FLAGS.save_font_array_file:
        with open(qnd.FLAGS.save_font_array_file, "w") as phile:
            json.dump(fonts.tolist(), phile)

    fonts -= fonts.mean()

    return fonts / np.sqrt((fonts ** 2).mean())


def def_font2char2word2sent2doc():
    adder = add_flags()
    classify = qndex.def_classify()
    word_array = qndex.nlp.def_word_array()

    def model(document, label=None, *, mode, key=None):
        return classify(
            font2char2word2sent2doc(
                document,
                words=word_array(),
                mode=mode,
                fonts=font_array(),
                **adder.flags),
            label,
            mode=mode,
            key=key,
            regularization_scale=qnd.FLAGS.regularization_scale)

    return model
