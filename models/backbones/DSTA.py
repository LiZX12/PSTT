import torch
from timm.models.resnet import resnet50
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

class DSTA(nn.Module):

    def __init__(self):
        super(DSTA, self).__init__()

        self.in_planes = 2048
        self.plances = 768
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.base = resnet50(True)
        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )
        self.base.global_pool = DSTA_X()
        self.base.fc = DSTA_X()

    def forward(self, x, pids=None, camid=None):
        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b * t, c, 16, 8)
        feat_map = self.down_channel(feat_map)
        return feat_map

class DSTA_X(nn.Module):

    def __init__(self):
        super(DSTA_X, self).__init__()

    def forward(self, x):
        return x
