import torch.nn as nn

class FewerChannelsAutoencoder(nn.Module):
    def __init__(self):
        super(FewerChannelsAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),   # 32x32x4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 16x16x4
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 16x16x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 8x8x8
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),  # 8x8x4
            nn.ReLU(),
            nn.Upsample(scale_factor=2),               # 16x16x4
            nn.Conv2d(4, 3, kernel_size=3, padding=1), # 16x16x3
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
