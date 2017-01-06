import functools

import extenteten as ex
import tensorflow as tf


def rd2sent2doc(document,
                word_embeddings,
                *,
                sentence_embedding_size,
                document_embedding_size,
                context_vector_size,
                save_memory=False):
    """
    word2sent2doc model lacking word embeddings as parameters
    """

    assert ex.static_rank(document) == 3
    assert ex.static_rank(word_embeddings) == 2

    embeddings_to_embedding = functools.partial(
        ex.bidirectional_embeddings_to_embedding,
        context_vector_size=context_vector_size)

    with tf.variable_scope("word2sent"):
        # word_embeddings.shape == (#batch * #sent * #word, emb_size)
        #                          if save_memory else
        #                          (vocab_size, emb_size)

        sentences = _flatten_document_into_sentences(document)

        sentence_embeddings = _restore_document_shape(
            embeddings_to_embedding(
                (_restore_sentence_shape(word_embeddings, sentences)
                 if save_memory else
                 tf.gather(word_embeddings, sentences)),
                sequence_length=ex.id_vector_to_length(sentences),
                output_size=sentence_embedding_size),
            document)

    with tf.variable_scope("sent2doc"):
        return embeddings_to_embedding(
            sentence_embeddings,
            sequence_length=ex.id_tensor_to_length(document),
            output_size=document_embedding_size)


@ex.func_scope()
def _flatten_document_into_sentences(document):
    return tf.reshape(document, [-1] + ex.static_shape(document)[2:])


@ex.func_scope()
def _restore_document_shape(sentences, document):
    return tf.reshape(
        sentences,
        [-1, ex.static_shape(document)[1]] + ex.static_shape(sentences)[1:])


@ex.func_scope()
def _restore_sentence_shape(words, sentences):
    return tf.reshape(
        words,
        [-1, ex.static_shape(sentences)[1]] + ex.static_shape(words)[1:])
