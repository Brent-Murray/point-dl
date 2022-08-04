# Adapted from https://github.com/pyg-team/pytorch_geometric/tree/master/examples

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    MLP,
    DynamicEdgeConv,
    EdgeConv,
    global_max_pool,
    knn_graph,
)


class DGCNN(torch.nn.Module):
    def __init__(self, num_classes, num_features, k=20):
        super().__init__()

        # Convolution Layers
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64, 64]), k)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64, 128]), k)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 128, 128, 256]), k)

        # Linear Layer
        self.lin1 = Linear(64 + 64 + 128 + 256, 1024)

        # MLP Layer
        self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5, batch_norm=False)

    def forward(self, data):
        # Get data values
        pos, batch = data.pos, data.batch

        # Network
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.lin1(torch.cat([x1, x2, x3, x4], dim=1))
        out = global_max_pool(out, batch)

        return self.mlp(out)
