import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat, reduce

MODEL_CONFIG = {
    
}


def create_regressor(model_config):
    pass


def create_model(model_config):
    feature_extractor = models.feature_extraction.create_feature_extractor(regressor_trained, target_layers)
    

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
        pass


class Distractor(nn.Module):
    def __init__(self, feature_extracter, readout_net, finilializer):
        super(Distractor, self).__init__()
        pass

    def forward(self, x):
        pass


class Regressor(nn.Module):
    def __init__(self, backbone_type='resnet18', pretrained=True, weights_path=None):
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
        regressor_extractor = []
        regressor_predictor = []
        
        flag = False
        for name, module in regressor.named_children():
            if not flag:
                regressor_extractor.append(module)
            else:
                regressor_predictor.append(module)
            if name == target_layer:
                flag = True
        
        self.regressor_extractor = nn.Sequential(*regressor_extractor)
        self.regressor_predictor = nn.Sequential(*regressor_predictor)
        self.distractor = distractor

    def forward(self, x):
        mask = self.regressor_extractor(x)
        x = self.regressor_extractor(x)
        x = x*mask + x
        x = self.regressor_predictor(x)
        return x
