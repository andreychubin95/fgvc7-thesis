from torch import nn
from torchvision import models

from .classifier import Classifier


class EfficientNetB7(nn.Module):
    def __init__(self, freeze_first_layers: bool = False):
        super(EfficientNetB7, self).__init__()
        self.net = self._get_model()
        if freeze_first_layers:
            self._freeze_layers()
        self.net.classifier = Classifier(self.net.classifier[1].in_features)

    @staticmethod
    def _get_model():
        return models.efficientnet_b7(pretrained=True)

    def _freeze_layers(self):
        n = len(self.net.features) - 3
        for i, layer in enumerate(self.net.features):
            if i < n:
                _ = layer.requires_grad_(False)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetB3(nn.Module):
    def __init__(self, freeze_first_layers: bool = False):
        super(EfficientNetB3, self).__init__()
        self.net = self._get_model()
        if freeze_first_layers:
            self._freeze_layers()
        self.net.classifier = Classifier(self.net.classifier[1].in_features)

    @staticmethod
    def _get_model():
        return models.efficientnet_b3(pretrained=True)

    def _freeze_layers(self):
        n = len(self.net.features) - 3
        for i, layer in enumerate(self.net.features):
            if i < n:
                _ = layer.requires_grad_(False)

    def forward(self, x):
        x = self.net(x)
        return x
