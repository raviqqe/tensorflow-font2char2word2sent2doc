import numpy as np
import qnd


__all__ = ['UNKNOWN_CHAR_INDEX', 'def_word_array']


UNKNOWN_CHAR_INDEX = 1


def def_word_array():
    qnd.add_flag("word_length", type=int, default=8)

    def word_array():
        word_array = np.zeros([len(qnd.FLAGS.words),
                               min(max(len(word) for word in qnd.FLAGS.words),
                                   qnd.FLAGS.word_length)],
                              np.int32)

        for i, word in enumerate(qnd.FLAGS.words):
            for j, char in enumerate(word[:qnd.FLAGS.word_length]):
                word_array[i, j] = (qnd.FLAGS.chars.index(char)
                                    if char in qnd.FLAGS.chars else
                                    UNKNOWN_CHAR_INDEX)

        return word_array

    return word_array
