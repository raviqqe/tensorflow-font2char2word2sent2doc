import numpy as np
import qnd


__all__ = ['NULL_INDEX', 'UNKNOWN_INDEX', 'def_word_array']


NULL_INDEX = 0
UNKNOWN_INDEX = 1


def def_word_array():
    qnd.add_flag('word_length', type=int, default=8)
    qnd.add_flag('save_word_array_file')

    def word_array():
        word_array = np.zeros([len(qnd.FLAGS.words),
                               min(max(len(word) for word in qnd.FLAGS.words),
                                   qnd.FLAGS.word_length)],
                              np.int32)

        for i, word in enumerate(qnd.FLAGS.words):
            for j, char in enumerate(word[:qnd.FLAGS.word_length]):
                word_array[i, j] = (qnd.FLAGS.chars.index(char)
                                    if char in qnd.FLAGS.chars else
                                    UNKNOWN_INDEX)

        word_array[NULL_INDEX, :] = NULL_INDEX
        word_array[UNKNOWN_INDEX, :] = NULL_INDEX
        word_array[UNKNOWN_INDEX, 0] = UNKNOWN_INDEX

        if qnd.FLAGS.save_word_array_file:
            np.savetxt(qnd.FLAGS.save_word_array_file,
                       word_array,
                       fmt='%d',
                       delimiter=',')

        return word_array

    return word_array
