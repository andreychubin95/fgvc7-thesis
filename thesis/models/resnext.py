from torch import nn
from torchvision import models

from .classifier import Classifier


class ResNeXt50(nn.Module):
    def __init__(self, freeze_first_layers: bool = False):
        super(ResNeXt50, self).__init__()
        self.resnext = self._get_model()
        if freeze_first_layers:
            self._freeze_layers()
        self.resnext.fc = Classifier(self.resnext.fc.in_features)

    @staticmethod
    def _get_model():
        return models.resnext50_32x4d(pretrained=True)

    def _freeze_layers(self):
        _ = self.resnext.conv1.requires_grad_(False)
        _ = self.resnext.bn1.requires_grad_(False)
        _ = self.resnext.relu.requires_grad_(False)
        _ = self.resnext.maxpool.requires_grad_(False)
        _ = self.resnext.layer1.requires_grad_(False)
        _ = self.resnext.layer2.requires_grad_(False)

    def forward(self, x):
        x = self.resnext(x)
        return x
