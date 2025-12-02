#!/usr/bin/env python3
import torch
import torch.nn as nn

# --------------------------
# Model
# --------------------------
class CNNFeaturePredictor(nn.Module):
    def __init__(self, S, M, P, num_classes):
        super().__init__()
        self.S = S
        self.M = M
        self.P = P

        # More stable feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.AdaptiveAvgPool2d((1, 50))
        )

        # More gradual classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 50 * S, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 1, self.M, self.P)
        x = self.feature_extractor(x)
        x = x.view(x.size(0)//self.S, -1)
        return self.classifier(x)


