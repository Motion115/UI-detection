import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=20):
        super(VGG, self).__init__()
        # configure VGG 16
        self.vgg_config_dict = {'VGG16':
                        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                        512, 'M', 512, 512, 512, 'M']
                }
        self.features = self._make_layers(self.vgg_config_dict['VGG16'])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(int(256 * 256 / 2), 768),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(768, 768)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, vgg_config_dict):
        layers = []
        in_channels = 3
        for v in vgg_config_dict:
            if v == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def get_embedding(self, x):
        x = self.features(x)
        x = self.embedding(x)
        return x