import extenteten as ex
import tensorflow as tf

from .rd2sent2doc import rd2sent2doc


def ar2word2sent2doc(document,
                     *,
                     words,
                     char_embeddings,
                     word_embedding_size,
                     context_vector_size,
                     save_memory=True,
                     **rd2sent2doc_hyperparams):
    """char2word2sent2doc model without character embeddings as parameters
    """

    assert ex.static_rank(document) == 3
    assert ex.static_rank(words) == 2
    assert ex.static_rank(char_embeddings) == 2

    with tf.variable_scope("char2word"):
        word_embeddings = ex.bidirectional_id_vector_to_embedding(
            tf.gather(words, ex.flatten(document)) if save_memory else words,
            char_embeddings,
            output_size=word_embedding_size,
            context_vector_size=context_vector_size,
            dynamic_length=True)

    return rd2sent2doc(document,
                       word_embeddings,
                       context_vector_size=context_vector_size,
                       save_memory=save_memory,
                       **rd2sent2doc_hyperparams)
