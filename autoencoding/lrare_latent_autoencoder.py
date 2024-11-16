import torch.nn as nn

class LargerLatentSpaceAutoencoder(nn.Module):
    def __init__(self):
        super(LargerLatentSpaceAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),   # 32x32x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 16x16x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # 8x8x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 8x8x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 4x4x32
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 4x4x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                # 8x8x16
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 8x8x8
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                # 16x16x8
            nn.Conv2d(8, 3, kernel_size=3, padding=1),  # 16x16x3
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
