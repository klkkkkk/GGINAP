import os
import torch
import numpy as np
import random
from config import *
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from dataloader import *
from networks.main_network import Network
from utils.utils import *
from tqdm import tqdm  # 进度条

def main(configs, fold_id, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    random.seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    train_loader = build_train_data(configs, fold_id=fold_id)
    # if configs.split == 'split20':
    #     valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    #emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    model = Network(configs).to(DEVICE)
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in params if '.bert' in n], 'lr': configs.bl, 'weight_decay': 0.01},
        {'params': [p for n, p in params if '.bert' not in n], 'lr': configs.lr, 'weight_decay': 0.01}
    ]

    optimizer = AdamW(optimizer_grouped_params, lr=configs.lr, no_deprecation_warning=True)

    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps

    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # training
    model.zero_grad()  # 清空梯度
    max_result_pair, max_result_emo, max_result_cau,max_result_pe,max_result_pc = None, None, None, None, None
    early_stop_flag = None

    for epoch in range(1, configs.epochs + 1):
        with tqdm(total=len(train_loader)) as pbar:
            for train_step, batch in enumerate(train_loader, 1):  # train_step从1数起
                model.train()  # 训练模式
                doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, bert_token_b, \
                bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, bert_clause_len_b, \
                graphs, emo_pos, cau_pos, y_emotion_category_b = batch

                pred_pair,pred_pair2, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                                    bert_clause_b, bert_clause_sep_b, bert_clause_len_b,
                                                                    doc_len_b, adj_b,y_mask_b,
                                                                    graphs, emo_pos, cau_pos,
                                                                    doc_id_b)
                loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
                loss_p, doc_couples_pred = model.loss_pair(pred_pair, emo_cau_pos, doc_couples_b)
                loss_p2, doc_couples_pred = model.loss_pair(pred_pair2, emo_cau_pos, doc_couples_b)
                mainloss = loss_e + loss_c + loss_p + loss_p2

                losses = (mainloss) / configs.gradient_accumulation_steps
                losses.backward()
                pbar.set_description("Epoch %d\tLoss %0.4f Le %0.4f Lc %0.4f Lp %0.4f" % (
                    epoch, losses, loss_e, loss_c, loss_p+loss_p2))
                pbar.update()
                if train_step % configs.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                # if train_step % 100 == 0:#每200轮输出一次信息
                #     print('epoch: {}, step: {}, loss: {}, {}, {}'.format(epoch, train_step, loss_e, loss_c, loss_p))

        with torch.no_grad():
            print('===== fold {} epoch {} TEST====='.format(fold_id, epoch))
            eval_emo, eval_cau, eval_pair,eval_pe,eval_pc = evaluate(configs, test_loader, model, tokenizer, epoch)
            print('pair - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(eval_pair[0], eval_pair[1], eval_pair[2]))
            print('emo - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(eval_emo[0], eval_emo[1], eval_emo[2]))
            print('cau - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(eval_cau[0], eval_cau[1], eval_cau[2]))
            print('pe - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(eval_pe[0], eval_pe[1], eval_pe[2]))
            print('pc - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(eval_pc[0], eval_pc[1], eval_pc[2]))

            if max_result_pair is None or eval_pair[0] > max_result_pair[0]:
                early_stop_flag = 1
                max_result_emo = eval_emo
                max_result_cau = eval_cau
                max_result_pair = eval_pair
                max_result_pe = eval_pe
                max_result_pc = eval_pc
            else:
                early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 10:
            print('===== fold {} early stop!====='.format(fold_id))
            break

    return max_result_emo, max_result_cau, max_result_pair, max_result_pe,max_result_pc

def evaluate(configs, test_loader, model, tokenizer, epoch):
    model.eval()
    all_emo, all_cau, all_pair,all_pe,all_pc = [0, 0, 0], [0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0]
    with tqdm(total=len(test_loader)) as pbar:
        for batch in test_loader:
            emo, cau, pair,pe,pc = evaluate_one_batch(configs, batch, model, tokenizer, epoch)
            pbar.set_description("Testing...")
            pbar.update()

            for i in range(3):
                all_emo[i] += emo[i]
                all_cau[i] += cau[i]
                all_pair[i] += pair[i]
                all_pe[i] += pe[i]
                all_pc[i] += pc[i]

    eval_emo = eval_func(all_emo)
    eval_cau = eval_func(all_cau)
    eval_pair = eval_func(all_pair)
    eval_pe = eval_func(all_pe)
    eval_pc = eval_func(all_pc)
    return eval_emo, eval_cau, eval_pair,eval_pe,eval_pc

def evaluate_one_batch(configs, batch, model, tokenizer, epoch):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, bert_token_b, \
    bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, bert_clause_len_b, \
    graphs, emo_pos, cau_pos, y_emotion_category_b = batch

    doc_id, doc_len, true_pairs = doc_id_b[0], doc_len_b[0], doc_couples_b[0]
    pair_set = []
    for i,pair in enumerate(true_pairs):
        if pair[0]!=pair[1]:
            for p in pair_set:
                if pair[0]==p[0] and pair[1]==p[1]:
                    true_pairs.pop(i)
                    break
            pair_set.append([pair[0],pair[1]])
    true_emo, true_cau = zip(*true_pairs)
    true_emo, true_cau = list(true_emo), list(true_cau)

    pred_pair, pred_pair2, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                                  bert_clause_b, bert_clause_sep_b, bert_clause_len_b,
                                                                  doc_len_b, adj_b,
                                                                  y_mask_b, graphs, emo_pos, cau_pos,
                                                                  doc_id_b)

    pred_pair = get_result(pred_pair)
    pred_pair2 = get_result(pred_pair2)
    pred_e = get_result(pred_e)
    pred_c = get_result(pred_c)

    pred_emo_final = []
    pred_cau_final = []
    pred_pair_final = []
    p_emo_final = []
    p_cau_final = []

    k = min(4, pred_e.size(0))

    _, emo_idx = torch.topk(pred_e, k=k, dim=0)
    _, cau_idx = torch.topk(pred_c, k=k, dim=0)
    pred_emo_final.append(int(emo_idx[0] + 1))
    pred_cau_final.append(int(cau_idx[0] + 1))
    for i, j in zip(*(emo_idx, cau_idx)):
        if pred_e[i] > 0.5 and i != emo_idx[0]:
            pred_emo_final.append(int(i + 1))
        if pred_c[j] > 0.5 and j != cau_idx[0]:
            pred_cau_final.append(int(j + 1))
        if len(pred_emo_final) > 3 or len(pred_cau_final) > 3:
            break

    weight = 0.6
    pred_pair_all = weight * pred_pair + (1 - weight) * pred_pair2

    _, idx = torch.topk(pred_pair_all, k=k, dim=0)
    pred_i = []
    pred_pair_final.append(emo_cau_pos[0][idx[0]])
    pred_i.append(idx[0])
    prob = [float(pred_pair[idx[0]])]
    p_emo_final.append(emo_cau_pos[0][idx[0]][0])
    p_cau_final.append(emo_cau_pos[0][idx[0]][1])

    for i in idx:
        threshold = 0.5
        if (pred_pair_all[i] > threshold) and i not in pred_i:
            sw = True
            if emo_cau_pos[0][i][1] in p_emo_final and emo_cau_pos[0][i][0] in p_cau_final:
                sw = False
            if emo_cau_pos[0][i][0]==emo_cau_pos[0][i][1] or sw==True:
                pred_pair_final.append(emo_cau_pos[0][i])
                pred_i.append(i)
                prob.append(float(pred_pair[i]))
                if emo_cau_pos[0][i][0]!=emo_cau_pos[0][i][1]:
                    p_emo_final.append(emo_cau_pos[0][i][0])
                    p_cau_final.append(emo_cau_pos[0][i][1])
        if len(pred_pair_final) > 4:
            break

    for pair in pred_pair_final:
        if pair[0] not in p_emo_final:
            p_emo_final.append(pair[0])
        if pair[1] not in p_cau_final:
            p_cau_final.append(pair[1])


    metric_e, metric_c, metric_p = \
        cal_metric(pred_emo_final, true_emo, pred_cau_final, true_cau, pred_pair_final, true_pairs, doc_len)
    metric_pe, metric_pc, metric_pp = \
        cal_metric(p_emo_final, true_emo, p_cau_final, true_cau, pred_pair_final, true_pairs, doc_len)

    return metric_e, metric_c, metric_p,metric_pe, metric_pc

def get_result(result):
    result = result.squeeze(0)
    result = torch.sigmoid(result)
    result = result.squeeze(-1)
    return result


if __name__ == '__main__':
    configs = Config()
    t = BertTokenizer.from_pretrained(configs.bert_cache_path)
    result_e, result_c, result_p,result_pe,result_pc = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for fold_id in range(1, 11):
        print('===== fold {} ====='.format(fold_id))
        metric_e, metric_c, metric_pair,metric_pe,metric_pc = main(configs, fold_id, t)
        print('Best pair result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(metric_pair[0], metric_pair[1], metric_pair[2]))
        print('Best emo  result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(metric_e[0], metric_e[1], metric_e[2]))
        print('Best cau  result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(metric_c[0], metric_c[1], metric_c[2]))
        print('Best pe  result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(metric_pe[0], metric_pe[1], metric_pe[2]))
        print('Best pc  result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(metric_pc[0], metric_pc[1], metric_pc[2]))

        for i in range(3):
            result_e[i] += metric_e[i]
            result_c[i] += metric_c[i]
            result_p[i] += metric_pair[i]
            result_pe[i] += metric_pe[i]
            result_pc[i] += metric_pc[i]
        print(
            'Current Average Pair - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_p[0] / fold_id, result_p[1] / fold_id,
                                                                        result_p[2] / fold_id))
        print('===== fold {} finished!====='.format(fold_id))

    for i in range(3):
        result_e[i] /= 10
        result_c[i] /= 10
        result_p[i] /= 10
        result_pe[i] /= 10
        result_pc[i] /= 10

    print('Average pair result- F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_p[0], result_p[1], result_p[2]))
    print('Average emo result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_e[0], result_e[1], result_e[2]))
    print('Average cau result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_c[0], result_c[1], result_c[2]))
    print('Average pe result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_pe[0], result_pe[1], result_pe[2]))
    print('Average pc result - F1:{:.4f}, P:{:.4f}, R:{:.4f}'.format(result_pc[0], result_pc[1], result_pc[2]))
