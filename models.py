import torch
import torch.nn as nn
from torchvision import models
import torch.nn.utils.weight_norm as weightNorm

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

res_dict = {
    "resnet18": models.resnet18
}

class ResBase(nn.Module):
    def __init__(self, res_name="resnet18"):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](weights=None)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x, return_intermediate=False):
        feats = []  # to collect intermediate features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if return_intermediate: feats.append(x)  # Layer 0

        x = self.layer1(x)
        if return_intermediate: feats.append(x)  # Layer 1

        x = self.layer2(x)
        if return_intermediate: feats.append(x)  # Layer 2

        x = self.layer3(x)
        if return_intermediate: feats.append(x)  # Layer 3

        x = self.layer4(x)
        if return_intermediate: feats.append(x)  # Layer 4

        fmap = x.clone()  # just before avgpool
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if return_intermediate:
            return x, feats  # feats is a list of feature maps per layer
        return x, fmap




class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim=512, bottleneck_dim=256, type="bn"):
        super().__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.type = type
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num=2, bottleneck_dim=256, type='wn'):  # 'wn' for weightNorm
        super().__init__()
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        return self.fc(x)


class FeatureTranslator(nn.Module):
    def __init__(self, feature_dim=512):
        super(FeatureTranslator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    def forward(self, x):
        return self.layers(x)

class FullTranslator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.net(x)


class ConditionalTranslator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, feat2, feat1):
        x = torch.cat([feat2, feat1], dim=1)
        return self.net(x)
