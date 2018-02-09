from __future__ import division
import sys
sys.path.append('..')
import time, os, cPickle
import dynet as dy
import models
from lib import Vocab, DataLoader
from config import Configurable
import argparse
import numpy as np

def parsing(string, tagging, parser, vocab):
    sent = string.split()
    tag = tagging.split()
    words = []
    tags = []
    for w in sent:
        words.append([vocab.word2id(w)])
    for t in tag:
        tags.append([vocab.tag2id(t)])
    words = np.array(words)
    tags = np.array(tags)
    outputs = parser.run(words, tags, isTrain=False)
    print(string.split())
    print(outputs[0])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--output_file', default='here')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    Parser = getattr(models, args.model)
    vocab = cPickle.load(open(config.load_vocab_path))
    parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp)
    parser.load(config.load_model_path)
    string = "The system as described above has its greatest application in an arrayed configuration of antenna elements ."
    tagging = "DT NN IN VBD IN VBZ PRP$ JJS NN IN DT VBD NNS IN NNS NNS ."
    parsing(string, tagging, parser, vocab)