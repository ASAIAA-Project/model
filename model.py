import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numbers
import functools

from einops import repeat
from torchvision.models import feature_extraction
from kornia import filters


def rgetattr(obj, attr, *args):
    return functools.reduce(lambda obj, attr: getattr(obj, attr, *args),
                            [obj] + attr.split('.'))


def create_model(model_config):
    regressor = Regressor(*model_config['regressor'])
    feature_extractor = feature_extraction.create_feature_extractor(
        regressor.backbone, model_config['feature_target_layer'])
    distractor = Distractor(feature_extractor,
                            Finializer(model_config['center_bias_weight']))
    return ASAIAANet(regressor, distractor, model_config['target_layer'])


class ReadoutNet(nn.Module):

    def __init__(self):
        super(ReadoutNet, self).__init__()
        pass

    def forward(self, x):
        pass


class Finializer(nn.Module):

    def __init__(self, center_bias_weight=1):
        super(Finializer, self).__init__()
        self.center_bias_weight = nn.Parameter(
            torch.Tensor([center_bias_weight]))

    def forward(self, x):
        x = 1 - filters.gaussian_blur2d(x, kernel_size=3, sigma=1.0)
        return x


class Distractor(nn.Module):

    def __init__(self, feature_extracter, finilializer):
        super(Distractor, self).__init__()
        self.feature_extracter = feature_extracter
        self.readout_net = ReadoutNet()
        self.finilializer = finilializer

    def forward(self, x):
        features = list(self.feature_extracter(x).values())
        for idx in len(features):
            if features[idx].shape[2] == 7:
                features = repeat(features[idx],
                                  'b c h w -> b c (h2 h) (w2 w)',
                                  h2=2,
                                  w2=2)
        feature = torch.cat(features, dim=1)
        x = self.readout_net(feature)
        x = self.finilializer(x)
        return x


class Regressor(nn.Module):

    def __init__(self,
                 backbone_type='resnet18',
                 pretrained=True,
                 weights_path=None):
        super(Regressor, self).__init__()
        self.backbone = models.__dict__[backbone_type](pretrained=pretrained)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)
        if weights_path:
            self.backbone.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.backbone(x)


class ASAIAANet(nn.Module):

    def __init__(self, regressor, distractor, target_layer):
        super(ASAIAANet, self).__init__()
        self.distractor = distractor
        self.target_layer = target_layer
        self.regressor = regressor

    def forward(self, x):
        mask = self.distractor(x)
        for node_name in feature_extraction.get_graph_node_names(
                self.regressor):
            x = rgetattr(self.regressor, node_name)(x)
            if node_name == self.target_layer:
                x = x * mask + x
        return x
