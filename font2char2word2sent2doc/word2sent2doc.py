import argtyp
import extenteten as ex
import tensorflow as tf
import qnd
import qndex

from .rd2sent2doc import rd2sent2doc


@ex.func_scope()
def word2sent2doc(document,
                  *,
                  word_space_size,
                  word_embedding_size,
                  **rd2sent2doc_hyperparams):
    assert ex.static_rank(document) == 3

    with tf.variable_scope("word_embeddings"):
        word_embeddings = tf.gather(
            ex.embeddings(id_space_size=word_space_size,
                          embedding_size=word_embedding_size,
                          name="word_embeddings"),
            ex.flatten(document))

    return rd2sent2doc(document,
                       word_embeddings,
                       save_memory=True,
                       **rd2sent2doc_hyperparams)


def add_flags():
    adder = qnd.FlagAdder()

    adder.add_flag("word_embedding_size", type=int, default=100)
    adder.add_flag("sentence_embedding_size", type=int, default=100)
    adder.add_flag("document_embedding_size", type=int, default=100)
    adder.add_flag("context_vector_size", type=int, default=100)

    return adder


def def_word2sent2doc():
    adder = add_flags()
    classify = qndex.def_classify()
    get_words = qndex.nlp.def_words()

    def model(document, label=None, *, mode, key=None):
        return classify(
            word2sent2doc(
                document,
                word_space_size=len(get_words()),
                **adder.flags),
            label,
            key=key,
            mode=mode)

    return model
