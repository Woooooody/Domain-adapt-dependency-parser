# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle

sys.path.append('..')
import dynet_config

dynet_config.set_gpu()
import dynet as dy
import numpy as np
import models
from lib import Vocab, DataLoader
from test import test
from config import Configurable
import random

import argparse

if __name__ == "__main__":
    np.random.seed(666)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    Parser = getattr(models, args.model)

    vocab = Vocab(config.wsj_file, config.pretrained_embeddings_file, config.min_occur_count)
    cPickle.dump(vocab, open(config.save_vocab_path, 'w'))
    parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers,
                    config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size,
                    config.mlp_rel_size, config.dropout_mlp, config.filter_size, config.domain_num)
    wsj = DataLoader(config.wsj_file, config.num_buckets_train, vocab, isTrain=True)
    answer = DataLoader(config.answers_file, config.num_buckets_train, vocab, isTrain=True, len_counter=wsj.len_counter)
    pc = parser.parameter_collection

    trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)

    data = []

    for i, item in enumerate([wsj,answer]):
        for words, tags, arcs, rels in item.get_batches(batch_size=config.train_batch_size, shuffle=True):
            data.append([words, tags, arcs, rels, i])

    random.shuffle(data)

    global_step = 0


    def update_parameters():
        trainer.learning_rate = config.learning_rate * config.decay ** (global_step / config.decay_steps)
        trainer.update()


    epoch = 0
    best_UAS = 0.
    history = lambda x, y: open(os.path.join(config.save_dir, 'valid_history'), 'a').write('%.2f %.2f\n' % (x, y))
    while global_step < config.train_iters:
        print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d' % (epoch,)
        epoch += 1
        lamb = (global_step * 1.0) / config.train_iters
        for words, tags, arcs, rels, domain_flag in data:
            num = int(words.shape[1] / 2)
            words_ = [words[:, :num], words[:, num:]]
            tags_ = [tags[:, :num], tags[:, num:]]
            arcs_ = [arcs[:, :num], arcs[:, num:]]
            rels_ = [rels[:, :num], rels[:, num:]]
            for step in xrange(2):
                dy.renew_cg()
                common_top_recur, private_top_recur, p_fs, p_bs = parser.run_lstm(words_[step], tags_[step])
                if domain_flag == 0:
                    arc_accuracy, rel_accuracy, overall_accuracy, parser_loss = parser.run_parser(words_[step],common_top_recur,private_top_recur, arc_targets=arcs_[step],rel_targets=rels_[step])
		    parser_loss = parser_loss * 0.5
                    parser_loss.backward()
                class_loss, class_accurate = parser.run_classifier(common_top_recur, words_[step], domain_flag)
                class_loss = lamb * class_loss * 0.5
                class_loss.backward()
                #lm_loss = parser.run_lm(p_fs, p_bs, words_[step])
                #lm_loss.backward()

                if domain_flag == 0:
                    loss = parser_loss+class_loss
                else:
                    loss = class_loss
                loss_value = loss.scalar_value()
                if domain_flag == 0:
                    sys.stdout.write("Step #%d: Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f\r\r" % (global_step, arc_accuracy, rel_accuracy, overall_accuracy, loss_value))
                    sys.stdout.flush()

            update_parameters()
            global_step += 1
            if global_step % config.validate_every == 0:
                print '\nTest on development set'
                LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file,
                    os.path.join(config.save_dir, 'valid_tmp'))
                history(LAS, UAS)
                if global_step > config.save_after and UAS > best_UAS:
                        best_UAS = UAS
                        parser.save(config.save_model_path)
