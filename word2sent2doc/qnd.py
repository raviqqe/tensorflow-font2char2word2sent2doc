import qnd

import .extenteten as ex
from .word2sent2doc import word2sent2doc


def words_file(filename):
    with open(filename) as phile:
        return phile.read().split()


def int_list(string):
    return [int(num) for num in string.split(",")]


def add_flags():
    adder = qnd.FlagAdder()

    qnd.add_required_flag("num_classes")
    qnd.add_required_flag("word_space_size")
    qnd.add_flag("word_embedding_size", type=int, default=100)
    qnd.add_flag("sentence_embedding_size", type=int, default=100)
    qnd.add_flag("document_embedding_size", type=int, default=100)
    qnd.add_flag("context_vector_size", type=int, default=100)
    qnd.add_flag("hidden_layer_sizes", type=int_list, default=[100])
    qnd.add_flag("dropout_keep_prob", type=float, default=1.0)

    return adder


def def_word2sent2doc():
    adder = add_flags()

    def model(document, labels):
        return word2sent2doc(
            document,
            labels,
            output_layer_size=(qnd.FLAGS.num_classes * ex.num_labels(labels)),
            **adder.flags)

    return model
