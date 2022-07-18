import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_scatter import scatter_max


class TransformerBlock(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Define Linear Layer
        self.lin_in = Lin(num_features, num_features)
        self.lin_out = Lin(num_classes, num_classes)

        # Define MLP Layer
        self.pos_nn = MLP([3, 64, num_classes], norm=None, plain_last=False)
        self.attn_nn = MLP([num_classes, 64, num_classes], norm=None, plain_last=False)

        # Define Transformer Layer
        self.transformer = PointTransformerConv(
            num_features, num_classes, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x
    
    
class TransitionDown(torch.nn.Module):
    def __init__(self, num_features, num_classes, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([num_features, num_classes], plain_last=False)

    def forward(self, x, pos, batch):
        # Furthest Point Sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # Find k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # Transformatoin of features through MLP
        x = self.mlp(x)

        # Max pool onto each cluster from knn
        x_out, _ = scatter_max(
            x[id_k_neighbor[1]], id_k_neighbor[0], dim_size=id_clusters.size(0), dim=0
        )

        # Keep only clusters and max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch
    
    
class PointTransformer(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim_model, k=16):
        super().__init__()
        self.k = k

        # Create dummy features if there are none
        num_features = max(num_features, 1)

        # First Block
        self.mlp_input = MLP([num_features, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(
            num_features=dim_model[0], num_classes=dim_model[0]
        )

        # Backbone Layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        # Append transformers_down and transition_down
        for i in range(len(dim_model) - 1):
            # transition_down
            self.transition_down.append(
                TransitionDown(
                    num_features=dim_model[i], num_classes=dim_model[i + 1], k=self.k
                )
            )

            # transformers_down
            self.transformers_down.append(
                TransformerBlock(
                    num_classes=dim_model[i + 1], num_features=dim_model[i + 1]
                )
            )

            # class score computation
            self.mlp_output = MLP([dim_model[-1], 64, num_classes], norm=None)

        def forward(self, data):
            x, pos, batch = data.x, data.pos, data.batch

            # Create dummy x if there are none
            if x is None:
                x = torch.ones((pos.shape[0], 1), device=pos.get_device())

            # First Block
            x = self.mlp_input(x)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformer_input(x, pos, edge_index)

            # Backbone
            for i in range(len(self.transformers_down)):
                x, pos, batch = self.transition_down[i](x, pos, batch=batch)

                edge_index = knn_graph(pos, k=self.k, batch=batch)
                x = self.transformers_down[i](x, pos, batch=batch)

            # Global Average Pooling
            x = global_mean_pool(x, batch)

            out = self.mlp_output(x)

            return out
        
        