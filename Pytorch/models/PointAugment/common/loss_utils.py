import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mat_loss(trans):
    d = trans.size()[1]
    I = troch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim(1,2)))
    
    return loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        # weighted_squared_errors = squared_errors * self.weights.unsqueeze(1)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss


def calc_loss(y_true, y_pred, weights):
    # loss = F.mse_loss(F.softmax(y_pred, dim=1), target=y_true)
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(F.softmax(y_pred, dim=1), y_true)
    
    return loss


def d_loss(y_true, y_pred, aug_y_pred, weights, w=0.5):
    """Loss function for the discriminator (classifier)"""
    LeakyReLU = nn.LeakyReLU(0.0)
    y_loss = calc_loss(y_true, y_pred, weights)  # loss for true
    aug_y_loss = calc_loss(y_true, aug_y_pred, weights)  # loss for augmented

    loss = (w * y_loss) + (w * aug_y_loss)  # caluclate loss with weight

    return loss


def g_loss(y_true, y_pred, aug_y_pred, data, aug, weights, lamb=2e-4):
    """Loss function for the generator (augmentor)"""
    pdist = nn.PairwiseDistance(p=1, keepdim=True)  # pairwise distance
    LeakyReLU = nn.LeakyReLU(0.0)  # leaky relu
    y_loss = calc_loss(y_true, y_pred, weights)  # loss for true
    aug_y_loss = calc_loss(y_true, aug_y_pred, weights)  # loss for augmented
    aug_pdist = pdist(data, aug).mul(lamb) # pairwise distance
    loss = LeakyReLU(y_loss - aug_y_loss + aug_pdist).mean() # final loss

    return loss