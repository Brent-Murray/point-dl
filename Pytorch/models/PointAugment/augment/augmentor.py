# Adapted from https://github.com/liruihui/PointAugment/blob/master/Augment/augmentor.py
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


def batch_quat_to_rotmat(q, out=None):
    B = q.size(0)

    if out is None:
        out = q.new_empty(B, 3, 3)

    # 2 / squared quaternion 2-norm
    leng = torch.sum(q.pow(2), 1)
    s = 2 / leng

    s_ = torch.clamp(leng, 2.0 / 3.0, 3.0 / 2.0)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)  # .mul(s_)
    out[:, 0, 1] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)  # .mul(s_)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)  # .mul(s_)

    return out, s_


class AugmentorRotation(nn.Module):
    def __init__(self, dim):
        super(AugmentorRotation, self).__init__()
        self.fc1 = nn.Linear(dim + 1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)

        return x, None
    
    
class AugmentorDisplacement(nn.Module):
    def __init__(self, dim):
        super(AugmentorDisplacement, self).__init__()
        self.conv1 = nn.Conv1d(
            dim + 1024 + 64, 1024, 1
        )
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 64, 1)
        self.conv4 = nn.Conv1d(64, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x
    
    
class Augmentor(nn.Module):
    def __init__(self, dim=1024, in_dim=3):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.rot = AugmentorRotation(self.dim)
        self.dis = AugmentorDisplacement(self.dim)

    def forward(self, group): # had to combine pt at noise into one
        data, noise = group
        B, C, N = data.size() # batch, channels, num points
        raw_pt = data[:, :3, :].contiguous()
        normal = data[:, 3:, :].transpose(1, 2).contiguous() if C > 3 else None
        
        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        feat_r = x.view(-1, 1024)
        feat_r = torch.cat([feat_r, noise], 1)
        rotation, scale = self.rot(feat_r)

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)
        feat_d = torch.cat([pointfeat, feat_d, noise_d], 1)
        displacement = self.dis(feat_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.0
        if p1 > possi:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()

        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt, normal], 1)
            

        return pt