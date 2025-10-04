import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=2, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, onehot_labels):
        x = z.view(z.size(0), self.latent_dim, 1, 1)
        y = onehot_labels.view(z.size(0), self.num_classes, 1, 1)
        return self.main(F.concat([x, y], dim=1))
