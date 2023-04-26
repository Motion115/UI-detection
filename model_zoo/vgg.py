import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn



# 定义网络
class VGG(nn.Module):
        # VGG16配置
    cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                    521, 'M', 512, 512, 512, 'M']}
    def __init__(self, net_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(self.cfg[net_name])

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32768, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 20)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)


