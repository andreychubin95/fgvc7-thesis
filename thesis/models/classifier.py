from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_features: int):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 4, bias=True)
        )

    def forward(self, x):
        return self.fc(x)
