#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==============================
# Model Components
# ==============================

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
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
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x1 = x1[:, :, :x2.size(2)]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==============================
# Performer Attention Modules
# ==============================

class FastAttention(nn.Module):
    """Fast Attention module based on Performer paper (FAVOR+)"""

    def __init__(self, dim, heads=8, dim_head=64, causal=False, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Random features for kernel approximation (fixed orthogonal initialization optional)
        self.register_buffer('random_features', torch.randn(dim_head, dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.kernel_fn = nn.ReLU()

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Apply kernel & scaling
        q = self.kernel_fn(q) * self.scale
        k = self.kernel_fn(k)

        # Ensure random_features on correct device
        if self.random_features.device != q.device:
            self.random_features = self.random_features.to(q.device)

        # Project via random features (Φ(x) = softmax(xR))
        def project(x):
            return torch.einsum('b h n d, d m -> b h n m', x, self.random_features)

        q_prime = project(q)
        k_prime = project(k)

        if self.causal:
            raise NotImplementedError("Causal attention not supported in genomics context.")
        else:
            attn_weights = torch.einsum('b h n m, b h l m -> b h n l', q_prime, k_prime)
            out = torch.einsum('b h n l, b h l d -> b h n d', attn_weights, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PerformerLayer(nn.Module):
    """Single Performer layer with fast attention and feed-forward"""

    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, causal=False, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FastAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention block
        x = x + self.attn(self.norm1(x))
        # Feed-forward block
        x = x + self.ff(self.norm2(x))
        return x


class FastAttentionBottleneck(nn.Module):
    """
    Bottleneck using Performer layers for linear-time attention.
    Khoromanski et al "Rethinking Attention with Performers" (ICLR 2021).
    """

    def __init__(
        self,
        in_channels,
        dim=384,
        heads=8,
        depth=2,
        dim_head=64,
        local_window_size=256,
        causal=False,
        dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.local_window_size = local_window_size

        # Projection to performer dimension
        self.proj_in = nn.Conv1d(in_channels, dim, kernel_size=1)
        self.norm_in = nn.LayerNorm(dim)

        # Performer layers
        self.layers = nn.ModuleList([
            PerformerLayer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                causal=causal,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Local context enhancement
        self.local_context = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        # Output projection
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Conv1d(dim, in_channels, kernel_size=1)

        # For adaptive processing of long sequences
        self.pool = nn.AdaptiveAvgPool1d(local_window_size)

    def forward(self, x):
        batch, C, orig_len = x.shape

        # Optionally pool long sequences
        if orig_len > self.local_window_size:
            x_pooled = self.pool(x)
        else:
            x_pooled = x

        # Project to latent dim
        x_proj = self.proj_in(x_pooled)  # [B, D, L]
        x_perm = x_proj.permute(0, 2, 1)  # [B, L, D]
        x_norm = self.norm_in(x_perm)

        # Apply performer layers
        for layer in self.layers:
            x_norm = layer(x_norm)

        # Add local context
        x_local = x_norm.permute(0, 2, 1)  # [B, D, L]
        x_local = self.local_context(x_local)
        x_local = x_local.permute(0, 2, 1)  # [B, L, D]
        x_norm = x_norm + x_local

        # Project back
        x_out = self.norm_out(x_norm)
        x_out = x_out.permute(0, 2, 1)  # [B, D, L]

        # Interpolate back if pooled
        if orig_len > self.local_window_size:
            x_out = F.interpolate(x_out, size=orig_len, mode='linear', align_corners=False)

        output = self.proj_out(x_out)
        return output + x  # Residual connection


# ==============================
# Main Model
# ==============================

class PerformerUNet(nn.Module):
    """UNet with Performer bottleneck (linear attention)"""

    def __init__(self, num_suppression_channels, bin_size):
        super().__init__()
        self.num_suppression = num_suppression_channels
        self.bin_size = bin_size
        total_inputs = 4 + num_suppression_channels
        print(f"\nInitializing PerformerUNet with {total_inputs} input channels...")

        # Encoder
        self.inc = DoubleConv(total_inputs, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Bottleneck
        self.bottleneck = FastAttentionBottleneck(
            in_channels=1024,
            dim=384,
            heads=8,
            depth=2,
            dim_head=64,
            local_window_size=256,
            causal=False
        )

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

        # Final binning and output
        x = F.avg_pool1d(x, kernel_size=self.bin_size, stride=self.bin_size)
        x = self.final_conv(x)
        return x.squeeze(1)


# ==============================
# Utilities
# ==============================

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def setup_model(num_suppression, bin_size, device):
    model = PerformerUNet(num_suppression, bin_size).to(device)
    model.apply(init_weights)
    return model


def setup_training(model, use_amp, device):
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    return scaler, optimizer, scheduler