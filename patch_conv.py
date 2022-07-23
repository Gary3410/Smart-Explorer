import torch
import torch.nn as nn
import torch.nn.functional as F

class patch_conv(nn.Module):
    def __init__(self, kernal_size=64):

        super(patch_conv, self).__init__()
        self.kernal_size = kernal_size
        keral = torch.ones((self.kernal_size, self.kernal_size))
        self.weight = nn.Parameter(data=keral, requires_grad=False)
    def forward(self, x1):
        channls = x1.size(1)
        keral_1 = self.weight.expand(channls, channls, self.kernal_size, self.kernal_size)
        x1 = F.conv2d(x1, keral_1)
        return x1