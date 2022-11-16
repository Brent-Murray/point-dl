import torch
from torch_geometric.nn import MLP


class Classifier(torch.nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model  # model
        self.num_classes = num_classes  # number of classes
        # MLP Layers
        self.mlp1 = MLP([1024, 1024, 1024, 512], dropout=0.5)
        self.mlp2 = MLP([512, 512, 512, 256], dropout=0.5)
        self.mlp3 = MLP(
            [256, 256, 256, self.num_classes], dropout=0.5, batch_norm=False
        )

    def forward(self, data):
        output = self.model(data)
        output = self.mlp1(output)
        output = self.mlp2(output)
        output = self.mlp3(output)

        return output