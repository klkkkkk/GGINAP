import pickle, json, decimal, math
import torch

def cal_metric(pred_emo, true_emo, pred_cau, true_cau, pred_pairs, true_pairs, doc_len):
    tp_e, fp_e, fn_e = 0, 0, 0
    tp_c, fp_c, fn_c = 0, 0, 0
    tp_p, fp_p, fn_p = 0, 0, 0
    for i in range(1, doc_len + 1):
        if i in pred_emo and i in true_emo:
            tp_e += 1
        elif i in pred_emo and i not in true_emo:
            fp_e += 1
        elif i not in pred_emo and i in true_emo:
            fn_e += 1
        if i in pred_cau and i in true_cau:
            tp_c += 1
        elif i in pred_cau and i not in true_cau:
            fp_c += 1
        elif i not in pred_cau and i in true_cau:
            fn_c += 1
    for pred_pair in pred_pairs:
        if pred_pair in true_pairs:
            tp_p += 1
        else:
            fp_p += 1
    for true_pair in true_pairs:
        if true_pair not in pred_pairs:
            fn_p += 1
    return [tp_e, fp_e, fn_e], [tp_c, fp_c, fn_c], [tp_p, fp_p, fn_p]

def eval_func(all_emo):
    precision_e = all_emo[0] / (all_emo[0] + all_emo[1] + 1e-6)
    recall_e = all_emo[0] / (all_emo[0] + all_emo[2] + 1e-6)
    f1_e = 2 * precision_e * recall_e / (precision_e + recall_e + 1e-6)
    return [f1_e, precision_e, recall_e]

def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js