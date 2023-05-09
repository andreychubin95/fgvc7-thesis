from torch import nn
from torchvision import models

from .classifier import Classifier


class Resnet50(nn.Module):
    def __init__(self, freeze_first_layers: bool = False):
        super(Resnet50, self).__init__()
        self.resnet = self._get_model()
        if freeze_first_layers:
            self._freeze_layers()
        self.resnet.fc = Classifier(self.resnet.fc.in_features)

    @staticmethod
    def _get_model():
        return models.resnet50(pretrained=True)

    def _freeze_layers(self):
        _ = self.resnet.conv1.requires_grad_(False)
        _ = self.resnet.bn1.requires_grad_(False)
        _ = self.resnet.relu.requires_grad_(False)
        _ = self.resnet.maxpool.requires_grad_(False)
        _ = self.resnet.layer1.requires_grad_(False)
        _ = self.resnet.layer2.requires_grad_(False)

    def forward(self, x):
        x = self.resnet(x)
        return x
