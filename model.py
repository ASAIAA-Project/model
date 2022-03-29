import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat, reduce


def create_model(args):
    pass


class ReadoutNet(nn.Module):
    def __init__(self):
        super(ReadoutNet, self).__init__()
        pass

    def forward(self, x):
        pass


class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layers):
        super(FeatureExtractor, self).__init__()
        pass

    def forward(self, x):
        pass


class Finializer(nn.Module):
    def __init__(self):
        super(Finializer, self).__init__()
        pass

    def forward(self, x):
        pass


class Distractor(nn.Module):
    def __init__(self, feature_extracter, readout_net):
        super(Distractor, self).__init__()
        pass

    def forward(self, x):
        pass


class Regressor(nn.Module):
    def __init__(self, backbone_type='resnet18', pretrained=True):
        super(Regressor, self).__init__()
        self.backbone = models.__dict__[backbone_type](pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.backbone(x)


class ASAIAANet(nn.Module):
    def __init__(self, regressor, distractor):
        super(ASAIAANet, self).__init__()
        pass

    def forward(self, x):
        pass
