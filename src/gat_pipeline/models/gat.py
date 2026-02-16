import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool as gmp


class GATNet(torch.nn.Module):
    def __init__(self, esm_embeds: int, n_heads: int, drop_prob: float, n_output: int):
        super().__init__()
        self.drop_prob = drop_prob
        self.gcn1 = GATConv(esm_embeds, esm_embeds, heads=n_heads, dropout=drop_prob)
        self.gcn2 = GATConv(esm_embeds * n_heads, esm_embeds, dropout=drop_prob)
        self.fc_g1 = nn.Linear(esm_embeds, 16)
        self.fc_g2 = nn.Linear(16, n_output)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        out = self.fc_g2(x)
        return out
