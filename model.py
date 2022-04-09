import torch
import torch.nn as nn
import torchvision.models as models
import functools

from einops import repeat
from torchvision.models import feature_extraction
from kornia import filters
from collections import OrderedDict

TARGET_LAYERS = [
    'layer3.1.conv1',  # 1, 256, 14, 14
    'layer3.1.conv2',  # 1, 256, 14, 14
    'layer3.0.conv2',  # 1, 256, 14, 14
    'layer4.0.conv2'
]  # 1, 512, 7, 7

FEATURE_CHANNELS_NUM = 256 + 256 + 256 + 512
FEATURE_H = 14
FEATURE_W = 14


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

    def __init__(self, feature_channels_num, feature_h, feature_w):
        super(ReadoutNet, self).__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('layernorm0',
                 nn.LayerNorm([feature_channels_num, feature_h, feature_w])),
                ('conv0', nn.Conv2d(FEATURE_CHANNELS_NUM,
                                    128, (1, 1),
                                    bias=True)),
                ('softplus0', nn.Softplus()),
                ('layernorm1', nn.LayerNorm([128, feature_h, feature_w])),
                ('conv1', nn.Conv2d(128, 16, (1, 1), bias=True)),
                ('softplus1', nn.Softplus()),
                ('conv2', nn.Conv2d(16, 1, (1, 1), bias=True)),
            ]))

    def forward(self, x):
        return self.net(x)


class Finializer(nn.Module):

    def __init__(self, center_bias_weight=1):
        super(Finializer, self).__init__()
        # self.center_bias_weight = nn.Parameter(
        #     torch.Tensor([center_bias_weight]))

    def forward(self, x):
        x = filters.gaussian_blur2d(x,
                                    kernel_size=[3, 3],
                                    sigma=[.75, .75],
                                    border_type='constant')
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
        self.backbone.eval()

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
