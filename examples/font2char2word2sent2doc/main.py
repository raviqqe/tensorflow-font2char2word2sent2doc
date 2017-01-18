#!/usr/bin/env python

import logging

import qnd
import font2char2word2sent2doc as f2c2w2s2d

import data


model = f2c2w2s2d.def_font2char2word2sent2doc()
read_file = data.def_read_file()
train_and_evaluate = qnd.def_train_and_evaluate()


def main():
    logging.getLogger().setLevel(logging.INFO)
    train_and_evaluate(model, read_file)


if __name__ == '__main__':
    main()
