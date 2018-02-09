# -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np

from lib import biLSTM, leaky_relu, bilinear, orthonormal_initializer, arc_argmax, rel_argmax, \
    orthonormal_VanillaLSTMBuilder


class BaseParser(object):
    def __init__(self, vocab,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 mlp_arc_size,
                 mlp_rel_size,
                 dropout_mlp,
                 filter_size,
                 domain_num
                 ):
        pc = dy.ParameterCollection()

        self._vocab = vocab
        self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
        self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs())
        self.tag_embs = pc.lookup_parameters_from_numpy(vocab.get_tag_embs(tag_dims))
        self.domain_num = domain_num
        self.lstm_hiddens = lstm_hiddens
        #   common LSTM
        self.cLSTM_builders = []
        f = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc)
        b = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc)
        self.cLSTM_builders.append((f, b))
        for i in xrange(lstm_layers - 1):
            f = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
            b = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
            self.cLSTM_builders.append((f, b))
        self.dropout_clstm_input = dropout_lstm_input
        self.dropout_clstm_hidden = dropout_lstm_hidden
        #   private LSTM
        self.pLSTM_builders = []
        f = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc)
        b = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc)
        self.pLSTM_builders.append((f, b))
        for i in xrange(lstm_layers - 1):
            f = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
            b = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
            self.pLSTM_builders.append((f, b))
        self.dropout_plstm_input = dropout_lstm_input
        self.dropout_plstm_hidden = dropout_lstm_hidden
        # ------Parser parameters---#
        mlp_size = mlp_arc_size + mlp_rel_size
        W = orthonormal_initializer(mlp_size, 4 * lstm_hiddens)
        self.mlp_dep_W = pc.parameters_from_numpy(W)
        self.mlp_head_W = pc.parameters_from_numpy(W)
        self.mlp_dep_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_head_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp

        self.arc_W = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init=dy.ConstInitializer(0.))
        self.rel_W = pc.add_parameters((vocab.rel_size * (mlp_rel_size + 1), mlp_rel_size + 1),
                                       init=dy.ConstInitializer(0.))
        # ------Domain Classifier parameters---#
        self.filter_size = filter_size
        self.filter = []
        for window_size in [3, 4, 5]:
            self.filter.append(pc.add_parameters((1, window_size, lstm_hiddens * 2, filter_size)))  # channel * window_size * emb_size * filter_num
        self.class_W = pc.add_parameters((self.domain_num, filter_size*3))

        # -------Language Model Parameters---------#
        # W_lm = orthonormal_initializer(lstm_hiddens, vocab.words_in_train)
        # self.lm_fw_W = pc.parameters_from_numpy(W_lm)
        # self.lm_bw_W = pc.parameters_from_numpy(W_lm)

        self._pc = pc

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for i in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched=True)
                tag_mask = dy.inputTensor(tag_mask, batched=True)
                ret.append((word_mask, tag_mask))
            return ret

        self.generate_emb_mask = _emb_mask_generator

    @property
    def parameter_collection(self):
        return self._pc

    def dynet_flatten_numpy(self, ndarray):
        return np.reshape(ndarray, (-1,), 'F')

    def run_lstm(self, word_inputs, tag_inputs, isTrain=True):
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]

        word_embs = [dy.lookup_batch(self.word_embs,
                                     np.where(w < self._vocab.words_in_train, w, self._vocab.UNK)) + dy.lookup_batch(
            self.pret_word_embs, w, update=False) for w in word_inputs]
        tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tag_inputs]

        if isTrain:
            emb_masks = self.generate_emb_mask(seq_len, batch_size)
            emb_inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                          zip(word_embs, tag_embs, emb_masks)]
        else:
            emb_inputs = [dy.concatenate([w, pos]) for w, pos in zip(word_embs, tag_embs)]

        common_top_input, c_fs, c_bs = biLSTM(self.cLSTM_builders,
                                              emb_inputs,
                                              batch_size,
                                              self.dropout_clstm_input if isTrain else 0.,
                                              self.dropout_clstm_hidden if isTrain else 0.)
        common_top_recur = dy.concatenate_cols(common_top_input)

        private_top_input, p_fs, p_bs = biLSTM(self.pLSTM_builders,
                                               emb_inputs,
                                               batch_size,
                                               self.dropout_plstm_input if isTrain else 0.,
                                               self.dropout_plstm_hidden if isTrain else 0.)
        private_top_recur = dy.concatenate_cols(private_top_input)

        if isTrain:
            common_top_recur = dy.dropout_dim(common_top_recur, 1, self.dropout_mlp)
            private_top_recur = dy.dropout_dim(private_top_recur, 1, self.dropout_mlp)

        return common_top_recur, private_top_recur, p_fs, p_bs

    def run_parser(self, word_inputs, common_top_recur, private_top_recur, arc_targets=None, rel_targets=None, isTrain=True):
        # inputs, targets: seq_len x batch_size

        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        num_tokens = int(np.sum(mask))
        top_recur = dy.concatenate([common_top_recur, private_top_recur])

        if isTrain or arc_targets is not None:
            mask_1D = self.dynet_flatten_numpy(mask)
            mask_1D_tensor = dy.inputTensor(mask_1D, batched=True)

        W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
        W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
        dep = leaky_relu(dy.affine_transform([b_dep, W_dep, top_recur]))
        head = leaky_relu(dy.affine_transform([b_head, W_head, top_recur]))
        if isTrain:
            dep = dy.dropout_dim(dep, 1, self.dropout_mlp)
            head = dy.dropout_dim(head, 1, self.dropout_mlp)

        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        W_arc = dy.parameter(self.arc_W)
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs=1,
                              bias_x=True, bias_y=False)
        # (#head x #dep) x batch_size

        flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len * batch_size)
        # (#head ) x (#dep x batch_size)

        arc_preds = arc_logits.npvalue().argmax(0)
        # seq_len x batch_size

        if isTrain or arc_targets is not None:
            arc_correct = np.equal(arc_preds, arc_targets).astype(np.float32) * mask
            arc_accuracy = np.sum(arc_correct) / num_tokens
            targets_1D = self.dynet_flatten_numpy(arc_targets)
            losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)
            arc_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens

        if not isTrain:
            arc_probs = np.transpose(
                np.reshape(dy.softmax(flat_arc_logits).npvalue(), (seq_len, seq_len, batch_size), 'F'))
        # #batch_size x #dep x #head

        W_rel = dy.parameter(self.rel_W)
        # dep_rel = dy.concatenate([dep_rel, dy.inputTensor(np.ones((1, seq_len),dtype=np.float32))])
        # head_rel = dy.concatenate([head_rel, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)
        # (#head x rel_size x #dep) x batch_size

        flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
        # (#head x rel_size) x (#dep x batch_size)

        partial_rel_logits = dy.pick_batch(flat_rel_logits,
                                           targets_1D if isTrain else self.dynet_flatten_numpy(arc_preds))
        # (rel_size) x (#dep x batch_size)

        if isTrain or arc_targets is not None:
            rel_preds = partial_rel_logits.npvalue().argmax(0)
            targets_1D = self.dynet_flatten_numpy(rel_targets)
            rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / num_tokens
            losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
            rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens

        if not isTrain:
            rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(),
                                                (self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
        # batch_size x #dep x #head x #nclasses

        if isTrain or arc_targets is not None:
            loss = arc_loss + rel_loss
            correct = rel_correct * self.dynet_flatten_numpy(arc_correct)
            overall_accuracy = np.sum(correct) / num_tokens

        if isTrain:
            return arc_accuracy, rel_accuracy, overall_accuracy, loss

        outputs = []

        for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = arc_argmax(arc_prob, sent_len, msk)
            rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_prob, sent_len)
            outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))

        if arc_targets is not None:
            return arc_accuracy, rel_accuracy, overall_accuracy, outputs
        return outputs

    def run_classifier(self, common_top_recur, word_inputs, domain_flag):
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        cnn_filter = []
        for filt in self.filter:
            cnn_filter.append(dy.parameter(filt))
        cnn_W = dy.parameter(self.class_W)

        cnn_input = dy.reshape(common_top_recur, (1, seq_len, 2 * self.lstm_hiddens), batch_size)
        # print(cnn_input.npvalue().shape)
        cnn_out_list = []
        for i in range(len(cnn_filter)):
            cnn_out = dy.conv2d(cnn_input, cnn_filter[i], [1, 1], is_valid=False)  # len*batch*filter_num
            # print(cnn_out.npvalue().shape)
            pool_out = dy.max_dim(cnn_out, d=1)
            # print(pool_out.npvalue().shape)
            pool_out = dy.reshape(pool_out, (self.filter_size,), batch_size)
            # print(pool_out.npvalue().shape)
            pool_out = dy.rectify(pool_out)
            cnn_out_list.append(pool_out)
        final_out = dy.concatenate(cnn_out_list)
        result = cnn_W * final_out
       
        predict = np.argmax(result.npvalue(), axis=0)
        # print(predict)
        cor = 0.
        for pre in predict:
            if int(pre) == domain_flag:
                cor += 1
        class_accurate = cor / batch_size

        target = [domain_flag] * batch_size  # [0,0,0,0]
        # print(result.npvalue().shape, np.array(target).shape)
        classes_loss = dy.pickneglogsoftmax_batch(result, target)
        class_loss = dy.sum_batches(classes_loss) / batch_size
        # print(class_loss.npvalue().shape)
        return class_loss, class_accurate

    def run_lm(self, p_fs, p_bs, word_inputs, tag_inputs):
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        num_tokens = int(np.sum(mask))
        flm_target, blm_target = [], []
        for sen in np.transpose(word_inputs):
            flm, blm = [], []
            blm.append(self._vocab.BOS)
            for i, word in enumerate(sen):
                if i != 0:
                    if word >= self._vocab.words_in_train:
                        word = self._vocab.UNK
                    flm.append(word)
                    blm.append(word)
            flm.append(self._vocab.EOS)
            flm_target.append(flm)
            blm_target.append(blm)

        fs = dy.concatenate_cols(p_fs)
        bs = dy.concatenate_cols(list(reversed(p_bs)))
        W_lm_fw, W_lm_bw = dy.transpose(dy.parameter(self.lm_fw_W), [1, 0]), dy.transpose(dy.parameter(self.lm_bw_W),[1, 0])
        f_out = W_lm_fw * fs
        b_out = W_lm_bw * bs
        f_out = dy.reshape(f_out, (self._vocab.words_in_train,), seq_len * batch_size)
        b_out = dy.reshape(b_out, (self._vocab.words_in_train,), seq_len * batch_size)
        targets_flm = self.dynet_flatten_numpy(flm_target)
        targets_blm = self.dynet_flatten_numpy(blm_target)
        f_losses = dy.pickneglogsoftmax_batch(f_out, targets_flm)
        b_losses = dy.pickneglogsoftmax_batch(b_out, targets_blm)
        flm_loss = dy.sum_batches(f_losses) / num_tokens
        blm_loss = dy.sum_batches(b_losses) / num_tokens
        return flm_loss + blm_loss

    def save(self, save_path):
        self._pc.save(save_path)

    def load(self, load_path):
        self._pc.populate(load_path)
