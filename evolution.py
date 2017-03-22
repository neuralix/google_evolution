import torch.nn as nn

__all__ = ['Evolution', 'evolution']

class Evolution(nn.Module):
    ch = [3,64,64,64,128,128,128,256,256,256,512,512,512]
    print(len(ch))

    def __init__(self, num_classes=10):
        super(Evolution, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(self.ch[0], self.ch[1], kernel_size=3, padding=1),
            nn.Conv2d(self.ch[1], self.ch[2], kernel_size=3, padding=1),
            nn.Conv2d(self.ch[2], self.ch[3], kernel_size=3, padding=1),
            nn.Conv2d(self.ch[3], self.ch[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[4]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[4], self.ch[5], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[5]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch[5]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[5], self.ch[6], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[6], self.ch[7], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[7]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[7], self.ch[8], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[8], self.ch[9], kernel_size=3, padding=1),
            nn.Conv2d(self.ch[9], self.ch[10], kernel_size=3, padding=1),
            nn.Conv2d(self.ch[10], self.ch[11], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[11]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch[11], self.ch[12], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch[12]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * self.ch[12], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 32 * self.ch[12])
        x = self.classifier(x)
        return x


def evolution(pretrained=False):
    model = Evolution()
    return model
