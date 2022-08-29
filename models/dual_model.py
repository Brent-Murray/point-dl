import torch

class DualModel(torch.nn.Module):
    def __init__(self, model_1, model_2, method="sum"):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.method = method # either "sum", "bilinear" or "concat"
        self.bilinear = torch.nn.Bilinear(1024, 1024, 1024) # bilinear layer
        self.linear = torch.nn.Linear(2048, 1024) # linear layer

    def forward(self, data):
        out1 = self.model_1(data)
        out2 = self.model_2(data)
        
        # Choose method of DualModel
        if self.method == "sum":
            output = out1.add(out2)
        if self.method == "bilinear":
            output = self.bilinear(out1, out2)
        if self.method == "concat":
            outs = torch.cat((out1, out2), 0)
            output = self.linear(outs)
        
        return output