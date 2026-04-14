import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from typing import Dict, List, Tuple
from torch_geometric.data import Data
import numpy as np
from gcn import *
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

def to_pyg_data(x_features: np.ndarray, adj_matrix: np.ndarray) -> Data:
    """
    将 NumPy 数组的特征和邻接矩阵转换为 PyTorch Geometric Data 对象。
    """
    x = torch.tensor(x_features, dtype=torch.float)
    from scipy.sparse import csr_matrix
    adj_sparse = csr_matrix(adj_matrix)
    adj_coo = adj_sparse.tocoo()
    row = adj_coo.row
    col = adj_coo.col
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super(GcnLayers, self).__init__()
        self.act = torch.nn.ELU()
        self.num_layers_num = num_layers_num
        self.g_net, self.bns, self.input_proj = self.create_net(n_in, n_h, self.num_layers_num)
        self.dropout = torch.nn.Dropout(p=dropout)

    def create_net(self, input_dim, hidden_dim, num_layers):
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else torch.nn.Identity()
        for i in range(num_layers):
            if i:
                nn = GCNConv(hidden_dim, hidden_dim, normalize=True)
            else:
                nn = GCNConv(input_dim, hidden_dim, normalize=True)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv)
            self.bns.append(bn)
        return self.convs, self.bns, self.input_proj

    def forward(self, x, edge_index):
        #x = self.input_proj(x)
        for i in range(self.num_layers_num):
            if i:
                graph_output = self.convs[i](graph_output, edge_index) + graph_output
            else:
                graph_output = self.convs[i](x, edge_index)
            graph_output = self.act(graph_output)
        return graph_output


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA


def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq


def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:, :k].shape)
    print(VT[:k, :].shape)
    res = U[:, :k].dot(np.diag(Sigma[:k]))
    return res


class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout, sample=None):
        super(PrePrompt, self).__init__()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if sample is not None:
            self.negative_sample = torch.tensor(sample, dtype=torch.int64).cuda()
        else:
            self.negative_sample = None
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        g = self.gcn(x, edge_index)
        loss = compareloss(g, self.negative_sample, temperature=1)
        return loss

    def embed(self, x, edge_index):
        g = self.gcn(x, edge_index)
        return g.detach() # n_h


def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp = temp.cuda()
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(edge_index, n):
    nodenum = edge_index.max().item() + 1
    adj_dict = {i: set() for i in range(nodenum)}
    for i, j in edge_index.T.tolist():
        adj_dict[i].add(j)
        adj_dict[j].add(i)

    res = np.zeros((nodenum, 1 + n), dtype=int)
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        neighbors = list(adj_dict[i])
        non_neighbors = np.setdiff1d(whole, neighbors)
        if len(neighbors) == 0:
            res[i][0] = i
        else:
            res[i][0] = neighbors[0]
        np.random.shuffle(non_neighbors)
        res[i][1:1 + n] = non_neighbors[:n]

    return res