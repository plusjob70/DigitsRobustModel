import torch.nn as nn
import torch


class RobustModel(nn.Module):
    def __init__(self):
        super(RobustModel, self).__init__()

        self.in_dim = 28 * 28 * 3
        self.out_dim = 10

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   # -> 26 x 26
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   # -> 24 x 24
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 96, 5, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   # -> 22 x 22
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 128, 5, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   # -> 20 x 20
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 160, 5, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   # -> 18 x 18
        self.fc = nn.Sequential(
            nn.Linear(10240, self.out_dim, bias=False),
            nn.BatchNorm1d(self.out_dim)
        )

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        conv5 = self.layer5(conv4)
        flat = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logit = self.fc(flat)

        return logit
