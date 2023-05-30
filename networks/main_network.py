import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from torch.nn.init import uniform_
from networks.HAN import *


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.bert_encoder = BertEncoder(configs)
        self.distance_emb = nn.Embedding(2 * configs.K + 1, configs.dis_emb_dim)
        self.pair_project = nn.Linear(configs.feat_dim * 2 + configs.dis_emb_dim, configs.feat_dim)
        self.pair_project2 = nn.Linear(configs.feat_dim * 2 + configs.dis_emb_dim, configs.feat_dim)

        self.HAN = HAN(
            meta_paths=[["EE"], ["PE"],
                        ["CC"], ["PC"]],
            in_size=configs.feat_dim,
            hidden_size=configs.feat_dim,
            num_heads=configs.han_heads,
            dropout=configs.dropout).to(DEVICE)
        self.HAN_pair = HAN(
            meta_paths=[["PEP"], ["PCP"]],
            in_size=configs.feat_dim * 2 + configs.dis_emb_dim,
            hidden_size=configs.feat_dim * 2,
            num_heads=configs.han_heads,
            dropout=configs.dropout).to(DEVICE)

        self.LayerNorm = nn.LayerNorm(configs.feat_dim, eps=configs.epsilon)

        self.pred_e_layer = nn.Linear(configs.feat_dim, 1)
        self.pred_c_layer = nn.Linear(configs.feat_dim, 1)
        self.pred_pair_layer0 = nn.Linear(configs.feat_dim * 3, configs.feat_dim)
        self.pred_pair_layer1 = nn.Linear(configs.feat_dim, 1)
        self.pred_pair_layer2 = nn.Linear(configs.feat_dim, 1)
        self.pred_pair_layer3 = nn.Linear(configs.feat_dim * 2, configs.feat_dim)

        self.dropout = nn.Dropout(configs.dropout)
        self.activation = nn.ReLU()

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, bert_clause_len_b,
                doc_len, adj, y_mask_b, graphs, emo_pos, cau_pos, doc_id_b, emotional_clauses=None):
        batch_size = len(doc_len)
        doc_sents_h = self.bert_encoder(bert_token_b, bert_masks_b, bert_segment_b, bert_clause_b, bert_clause_sep_b,
                                        bert_clause_len_b, doc_len)

        word_embedding_by_sentence = self.get_word_embedding_bert(doc_sents_h, bert_clause_b, bert_clause_sep_b,
                                                                  batch_size, doc_len)
        doc_sents_h = torch.stack(
            [torch.mean(word_embedding_by_sentence[i], dim=0) for i in range(len(word_embedding_by_sentence))])
        sentence = doc_sents_h

        sentence_emo = sentence
        sentence_cau = sentence.clone()

        tmp_sentence = sentence.split(doc_len.tolist())
        pair = [
            torch.cat([tmp_sentence[i].index_select(0, emo_pos[i]), tmp_sentence[i].index_select(0, cau_pos[i])], -1)
            for i in range(batch_size)]
        pair = torch.cat(pair)

        position_tag = []
        for i in range(batch_size):
            for j in range(len(emo_pos[i])):
                position_tag.append(int(self.configs.K + emo_pos[i][j] - cau_pos[i][j]))
        position_tag = torch.LongTensor(position_tag).to(DEVICE)
        distance_rep = self.distance_emb(position_tag)
        pair = torch.cat((pair, distance_rep), dim=-1)
        pair = self.pair_project(pair)

        emo_cau_list = []
        emo_cau_cnt = 0
        for i in range(batch_size):
            sen_emo = sentence_emo[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            sen_cau = sentence_cau[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            emo_cau_list.append(
                torch.cat([sen_emo.index_select(0, emo_pos[i]), sen_cau.index_select(0, cau_pos[i])], dim=1))
            emo_cau_cnt += doc_len[i]
        emo_cau = torch.cat(emo_cau_list)
        pair0 = pair
        couples_pred2 = self.pred_pair_layer1(pair0)

        bg = dgl.batch(graphs).to(DEVICE)

        features = {"emo": sentence_emo, "cau": sentence_cau, "pair": pair}

        new_features = self.HAN(bg, features, doc_len)
        emo = new_features[0]
        cau = new_features[1]
        pair = new_features[2]

        pred_e, pred_c = self.pred_e_layer(emo), self.pred_c_layer(cau)

        emo_cau_list = []
        emo_cau_cnt = 0
        for i in range(batch_size):
            sen_emo = emo[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            sen_cau = cau[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            emo_cau_list.append(
                torch.cat([sen_emo.index_select(0, emo_pos[i]), sen_cau.index_select(0, cau_pos[i])], dim=1))
            emo_cau_cnt += doc_len[i]
        emo_cau = torch.cat(emo_cau_list)
        emo_cau = torch.cat((emo_cau, distance_rep), dim=-1)

        features = {"emo": emo, "cau": cau, "pair": emo_cau}

        new_features = self.HAN_pair(bg, features, doc_len)
        pair = new_features[2]

        couples_pred = self.pred_pair_layer2(self.activation(self.pred_pair_layer3(pair)))

        emo_cau_pos = []
        for i in range(batch_size):
            emo_cau_pos_i = []
            for emo, cau in zip(emo_pos[i], cau_pos[i]):
                emo_cau_pos_i.append([int(emo + 1), int(cau + 1)])
            emo_cau_pos.append(emo_cau_pos_i)
        return couples_pred.squeeze(-1), couples_pred2.squeeze(-1), emo_cau_pos, pred_e.squeeze(-1), pred_c.squeeze(-1)

    def get_word_embedding_bert(self, bert_output, bert_clause_b, bert_clause_sep_b, batch_size, doc_len):
        assert bert_clause_b.size() == bert_clause_sep_b.size()
        hidden_state = bert_output[0]
        word_embedding_by_sentence = []
        for i in range(batch_size):
            for j in range(doc_len[i]):
                cls = bert_clause_b[i][j]
                sep = bert_clause_sep_b[i][j]
                word_embedding_by_sentence.append(hidden_state[i][cls + 1: sep])

        return word_embedding_by_sentence

    def loss_pair(self, couples_pred, emo_cau_pos, doc_couples, test=False):
        couples_true = []
        batch_size = len(emo_cau_pos)
        for i in range(batch_size):
            couples_num = len(emo_cau_pos[i])
            true_indices = [emo_cau_pos[i].index(x) for x in doc_couples[i] if abs(x[0] - x[1]) <= self.configs.K]
            temp = torch.zeros(couples_num)
            temp[true_indices] = 1.
            couples_true.append(temp)
        couples_true = torch.cat(couples_true)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_couple = criterion(couples_pred, couples_true)
        doc_couples_pred = []
        if test:
            couples_pred = couples_pred.split([len(x) for x in emo_cau_pos])
            for i in range(batch_size):
                couples_pred_i = couples_pred[i]
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 1
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=1, dim=0)
                # (位置，网络输出的得分)
                doc_couples_pred_i = [(emo_cau_pos[i][idx], couples_pred_i[idx].tolist()) for idx in k_idx]
                doc_couples_pred.append(doc_couples_pred_i)
        return loss_couple, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask, test=False):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)

        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)
        if test:
            return loss_e, loss_c, pred_e, pred_c, true_e, true_c
        return loss_e, loss_c


class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(hidden_size, 1)
        self.fc_query = nn.Linear(hidden_size, 1)

    def forward(self, text, text_mask, text_seg, bert_clause_b, bert_clause_sep_b, bert_clause_len_b, doc_len):
        hidden_states = self.bert(input_ids=text.to(DEVICE),
                                  attention_mask=text_mask.to(DEVICE),
                                  token_type_ids=text_seg.to(DEVICE))
        return hidden_states