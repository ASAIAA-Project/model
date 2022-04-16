import torch
import torch.nn as nn
import torchvision.models as models

from einops import repeat
from torchvision.models import feature_extraction
from kornia import filters
from collections import OrderedDict


def create_ASAIAANet(model_config):
    regressor = Regressor(model_config.backbone_type, model_config.pretrained,
                          model_config.weight_path)
    feature_extractor = feature_extraction.create_feature_extractor(
        regressor.backbone, model_config.feature_target_layer)
    distractor = Distractor(
        feature_extractor, Finializer(model_config.center_bias_weight),
        ReadoutNet(model_config.feature_channels_num, model_config.feature_h,
                   model_config.feature_w))
    return ASAIAANet(regressor, distractor, model_config.distracting_block)


class ReadoutNet(nn.Module):

    def __init__(self, feature_channels_num, feature_h, feature_w):
        super(ReadoutNet, self).__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('layernorm0',
                 nn.LayerNorm([feature_channels_num, feature_h, feature_w])),
                ('conv0', nn.Conv2d(feature_channels_num,
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

    def __init__(self, center_bias_weight=1, kernel_size=3, sigma=1):
        super(Finializer, self).__init__()
        # self.center_bias_weight = nn.Parameter(
        #     torch.Tensor([center_bias_weight]))
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        x = filters.gaussian_blur2d(
            x,
            kernel_size=[self.kernel_size, self.kernel_size],
            sigma=[self.sigma, self.sigma],
            border_type='constant')
        return x


class Distractor(nn.Module):

    def __init__(self, feature_extracter, finilializer, readout_net):
        super(Distractor, self).__init__()
        self.feature_extracter = feature_extracter
        self.readout_net = readout_net
        self.finilializer = finilializer

        self.feature_extracter.eval()
        for param in self.feature_extracter.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = list(self.feature_extracter(x).values())
            for idx in range(len(features)):
                if features[idx].shape[2] == 7:
                    features[idx] = repeat(features[idx],
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
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

        if weights_path:
            self.backbone.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.backbone(x)


class ASAIAANet(nn.Module):

    def __init__(self, regressor, distractor, target_block):
        super(ASAIAANet, self).__init__()
        self.distractor = distractor
        self.target_block = target_block
        self.regressor = regressor

    def forward(self, x):
        mask = self.distractor(x)
        for name, block in self.regressor.backbone.named_children():
            if name == 'fc':
                x = x.view(-1, 512)
                x = block(x)
            else:
                x = block(x)
                if name == self.target_block:
                    x = x * mask + x
        return x
