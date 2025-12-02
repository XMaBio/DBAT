#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_model(num_suppression, bin_size, device):
    model = TransUNet(num_suppression, bin_size).to(device)

    model.apply(init_weights)

    return model


def setup_training(model, use_amp, device):
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp) #
    else:
        scaler = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    return scaler, optimizer, scheduler



class TransUNet(nn.Module):
    """TransUNet with skip connections"""

    def __init__(self, num_suppression_channels, bin_size):
        super().__init__()
        self.num_suppression = num_suppression_channels
        self.bin_size = bin_size
        total_inputs = 4 + num_suppression_channels
        print(f"\nInitializing TransUNet with {total_inputs} input channels...")

        # Encoder
        self.inc = DoubleConv(total_inputs, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Transformer Bottleneck
        self.bottleneck = TransformerBottleneck(1024, 384, max_tokens=256)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final output
        self.final_conv = nn.Conv1d(64, 1, kernel_size=1)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params / 1e6:.2f}M parameters")

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        x5 = self.bottleneck(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final processing
        x = F.avg_pool1d(
            x, kernel_size=self.bin_size, stride=self.bin_size
        )
        x = self.final_conv(x)
        return x.squeeze(1)



class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool, then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connections"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x1 = x1[:, :, :x2.size()[2]]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class TransformerBottleneck(nn.Module):
    """Transformer Bottleneck with Adaptive Pooling for Long Sequences"""

    def __init__(self, in_channels, embed_dim, max_tokens=256, num_layers=2, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        self.proj_in = nn.Linear(in_channels, embed_dim)
        self.proj_out = nn.Linear(embed_dim, in_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        batch, C, orig_len = x.size()
        x_pooled = F.adaptive_avg_pool1d(x, self.max_tokens)
        x_pooled = x_pooled.permute(0, 2, 1)
        x_pooled = self.proj_in(x_pooled)
        x_pooled = x_pooled + self.pos_embed

        for block in self.transformer_blocks:
            x_pooled = block(x_pooled)

        x_pooled = self.proj_out(x_pooled)
        x_pooled = x_pooled.permute(0, 2, 1)
        x = F.interpolate(x_pooled, size=orig_len, mode='linear', align_corners=False)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Layer Normalization and Residual Connections"""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x



def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

