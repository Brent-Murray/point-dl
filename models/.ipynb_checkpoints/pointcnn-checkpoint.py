import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import XConv, fps, global_mean_pool


class PointCNN(nn.Module):
    def __init__(self, numfeatures, numclasses):
        super().__init__()
        self.numfeatures = numfeatures
        self.numclasses = numclasses

        # First XConv layer
        self.conv1 = XConv(
            self.numfeatures, 48, dim=3, kernel_size=8, hidden_channels=32
        )

        # Second XConv layer
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2
        )

        # Third XConv layer
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2
        )

        # Fourth XConv layer
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2
        )

        # Multilayer Perceptrons (MLPs) at the end of the PointCNN
        self.lin1 = nn.Linear(384, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(
            128, self.numclasses
        )  # change last value for number of classes

    def forward(self, data):
        # Get pos and batch
        pos, batch = data.pos, data.batch

        # Get x
        x = data.x if self.numfeatures else None

        # First XConv with no features
        x = F.relu(self.conv1(x, pos, batch))
        # x = torch.nn.ReLU(self.conv1(x, pos, batch))

        # Farthest point sampling, keeping only 37.5%
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        # Second XConv
        x = F.relu(self.conv2(x, pos, batch))

        # Farthest point samplling, keepiong only 33.4%
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        # Two additional XConvs
        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        # Pooling batch-elements together
        # Each tree is described in one single point with 384 features
        x = global_mean_pool(x, batch)

        # MLPs at the end with ReLU
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        # Dropout: Set randomly to value of zero
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.dropout(x, p=0.5, training=True)
        return self.lin3(x)