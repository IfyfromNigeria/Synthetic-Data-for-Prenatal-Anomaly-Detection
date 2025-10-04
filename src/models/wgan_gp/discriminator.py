import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, num_classes=2, img_channels=1, img_size=64):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.main = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(16, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(256 * (img_size // 8) ** 2, 1))

    def forward(self, img, onehot_labels):
        y = onehot_labels.view(img.size(0), self.num_classes, 1, 1).repeat(1,1,self.img_size,self.img_size)
        x = torch.cat([img, y], dim=1)
        return self.out(self.main(x))
