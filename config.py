import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TORCH_SEED = 2022
DATA_DIR = 'data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE = 'fold%s_test.json'
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'

class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = 'pretrained_model/bert-base-chinese'

        # hyper parameter
        self.feat_dim = 768
        self.K = 3
        self.epsilon = 1e-8
        self.max_doc_len = 73
        self.max_token_len = 512
        self.dropout = 0
        self.layers = 1
        self.epochs = 30
        self.lr = 9e-6
        self.bl = 1.3e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 1
        self.l2 = 1e-5
        self.l2_bert = 0.
        self.warmup_proportion = 0.02
        self.adam_epsilon = 1e-8

        # gnn
        self.feat_dim = 768
        self.gnn_dims = '192'
        self.att_heads = '4'

        #other
        self.max_distance = 4
        self.dis_emb_dim = 200
        self.score_dim = 100
        self.han_heads = [1]

        #network
        self.sub_layers = 1
        self.pair_layers = 1