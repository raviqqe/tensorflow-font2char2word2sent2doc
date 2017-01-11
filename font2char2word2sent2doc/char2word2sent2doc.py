import extenteten as ex
import tensorflow as tf

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
