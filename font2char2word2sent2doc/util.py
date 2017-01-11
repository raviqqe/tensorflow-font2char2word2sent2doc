import numpy as np
import qnd


UNKNOWN_CHAR_INDEX = 1


def word_array():
    word_array = np.zeros([len(qnd.FLAGS.words),
                           max(len(word) for word in qnd.FLAGS.words)],
                          np.int32)

    for i, word in enumerate(qnd.FLAGS.words):
        for j, char in enumerate(word):
            word_array[i, j] = (qnd.FLAGS.chars.index(char)
                                if char in qnd.FLAGS.chars else
                                UNKNOWN_CHAR_INDEX)

    return word_array
