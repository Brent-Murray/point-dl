import torch

class Sum(torch.nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, data):
        out1 = self.model_1(data)
        out2 = self.model_2(data)
        
        output = out1.add(out2)
        
        return output
    
    
class Bilinear(torch.nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.bilinear = torch.nn.Bilinear(1024, 1024, 1024) # bilinear layer
        
    def forward(self, data):
        out1 = self.model_1(data)
        out2 = self.model_2(data)
        
        output = self.bilinear(out1, out2)
        
        return output
    
    
class Concat(torch.nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.linear = torch.nn.Linear(2048, 1024) # linear layer
        
    def forward(self, data):
        out1 = self.model_1(data)
        out2 = self.model_2(data)
        
        output = self.linear(torch.cat([out1, out2], dim=1))
        
        return output