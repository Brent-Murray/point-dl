import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio  # ratio
        self.r = r  # radius
        self.conv = PointConv(nn, add_self_loops=False)  # point conv

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)  # furthest point sampling
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )  # https://pytorch-geometric.readthedocs.io/en/1.3.1/modules/nn.html#torch_geometric.nn.pool.radius
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)  # new x after PointConv
        pos, batch = pos[idx], batch[idx]  # new pos, batch based on idx

        return x, pos, batch
    
    
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn  # neural net

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)  # max pooling
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)

        return x, pos, batch
    
    
class Net(torch.nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.num_classes = num_classes

        # SAModules
        # SAModule(ratio, r, nn)
        # GlobalSAModule(nn)
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # MLP
        # MLP: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.MLP
        self.mlp = MLP(
            [1024, 512, 256, self.num_classes], dropout=0.5, batch_norm=False
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Return Log Sofmax of output
        # return self.mlp(x).log_softmax(dim=-1)
        return self.mlp(x)