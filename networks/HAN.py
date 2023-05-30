"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import GATConv
from config import DEVICE


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        self.project = nn.ModuleList()
        for i in range(3):
            self.project.append(nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
            ))

    def forward(self, semantic, doc_len, h):
        res = []
        for i, z in enumerate(semantic):
            if len(z) == 0:
                if i==0:
                    res.append(h['emo'])
                elif i==1:
                    res.append(h['cau'])
                else:
                    res.append(h['pair'])
                continue
            w = self.project[i](z)#.mean(0)
            beta = torch.softmax(w, dim=1)
            #beta = beta.expand((z.shape[0],) + beta.shape)
            res.append((beta * z).sum(1))

        #beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return res  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    (in_size,in_size),
                    out_size,
                    layer_num_heads,
                    dropout,
                    0,
                    residual=True,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads,hidden_size=out_size * layer_num_heads)
        self.LayerNorm = nn.LayerNorm(out_size * layer_num_heads, eps=1e-8)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, doc_len, node_num=None):
        semantic_embeddings_emo = []
        semantic_embeddings_cau = []
        semantic_embeddings_pair = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]#取出该元路径对应的子图
            feat = [meta_path[0][0],meta_path[-1][-1]]
            for j in range(len(feat)):
                f = feat[j]
                if f == 'E':
                    f = 'emo'
                elif f == 'C':
                    f = 'cau'
                else:
                    f = 'pair'
                if j == 0:
                    src_feat = f
                else:
                    dst_feat = f

            hidden_emb = self.gat_layers[i](new_g, (h[src_feat], h[dst_feat])).flatten(1)
            #hidden_emb = self.LayerNorm(hidden_emb)

            eval('semantic_embeddings_' + dst_feat + '.append(hidden_emb)')
        if len(semantic_embeddings_emo) > 0:
            semantic_embeddings_emo = torch.stack(semantic_embeddings_emo, dim=1)  # (N, M, D * K)
        if len(semantic_embeddings_cau) > 0:
            semantic_embeddings_cau = torch.stack(semantic_embeddings_cau, dim=1)  # (N, M, D * K)
        if len(semantic_embeddings_pair) > 0:
            semantic_embeddings_pair = torch.stack(semantic_embeddings_pair, dim=1)  # (N, M, D * K)

        semantic_embeddings = [semantic_embeddings_emo, semantic_embeddings_cau,
                               semantic_embeddings_pair]
        return self.semantic_attention(semantic_embeddings, doc_len, h)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        #self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h, doc_len, node_num=None):
        for i,gnn in enumerate(self.layers):
            if i > 0:
                h = {'emo': h[0],'cau': h[1],'pair': h[2]}
            h = gnn(g, h, doc_len)

        return h
