import qnd
import argtyp

from . import extenteten as ex
from .word2sent2doc import word2sent2doc


def add_flags():
    adder = qnd.FlagAdder()

    adder.add_required_flag("word_space_size", type=int)
    adder.add_flag("word_embedding_size", type=int, default=100)
    adder.add_flag("sentence_embedding_size", type=int, default=100)
    adder.add_flag("document_embedding_size", type=int, default=100)
    adder.add_flag("context_vector_size", type=int, default=100)

    return adder


def def_classify():
    qnd.add_required_flag("num_classes", type=int)
    qnd.add_flag("hidden_layer_sizes", type=argtyp.int_list, default=[100])
    qnd.add_flag("dropout_keep_prob", type=float, default=1)

    def classify(feature, label):
        return ex.classify(
            ex.mlp(
                feature,
                layer_sizes=[
                    *qnd.FLAGS.hidden_layer_sizes,
                    ex.num_logits(ex.num_labels(label), qnd.FLAGS.num_classes)],
                dropout_keep_prob=qnd.FLAGS.dropout_keep_prob),
            label,
            binary=(qnd.FLAGS.num_classes == 2))

    return classify


def def_word2sent2doc():
    adder = add_flags()
    classify = def_classify()

    def model(document, label):
        return classify(word2sent2doc(document, **adder.flags), label)

    return model
