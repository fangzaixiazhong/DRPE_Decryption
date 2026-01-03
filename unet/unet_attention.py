import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def CBR(in_channels, out_channels, dropout_p=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout2d(p=dropout_p))
    return nn.Sequential(*layers)

class FeatureNoise(nn.Module):
    def __init__(self, sigma=0.02, apply_prob=0.3):
        super(FeatureNoise, self).__init__()
        self.sigma = sigma
        self.apply_prob = apply_prob

    def forward(self, x):
        if self.training and random.random() < self.apply_prob:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        else:
            return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B,1,H,W]
        return x * psi  # gated features

class UNetAttention(nn.Module):
    def __init__(self):
        super(UNetAttention, self).__init__()

        self.noise1 = FeatureNoise(sigma=0.025)
        self.noise2 = FeatureNoise(sigma=0.02)
        self.noise3 = FeatureNoise(sigma=0.015)
        self.noise4 = FeatureNoise(sigma=0.01)

        self.encoder1 = nn.Sequential(CBR(1, 64, dropout_p=0.02), CBR(64, 64, dropout_p=0.02))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(CBR(64, 128, dropout_p=0.04), CBR(128, 128, dropout_p=0.04))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(CBR(128, 256, dropout_p=0.06), CBR(256, 256, dropout_p=0.06))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = nn.Sequential(CBR(256, 512, dropout_p=0.08), CBR(512, 512, dropout_p=0.08))
        self.pool4 = nn.MaxPool2d(2, 2)


        self.bottom = nn.Sequential(CBR(512, 1024, dropout_p=0.10), CBR(1024, 1024, dropout_p=0.10))

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = nn.Sequential(CBR(1024, 512, dropout_p=0.08), CBR(512, 256, dropout_p=0.08))

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder3 = nn.Sequential(CBR(512, 256, dropout_p=0.04), CBR(256, 128, dropout_p=0.04))

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder2 = nn.Sequential(CBR(256, 128, dropout_p=0.02), CBR(128, 64, dropout_p=0.02))

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.decoder1 = nn.Sequential(CBR(128, 64, dropout_p=0.01), CBR(64, 64, dropout_p=0.01))

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.encoder1(x)           # 64
        p1 = self.pool1(e1)
        p1 = self.noise1(p1)

        e2 = self.encoder2(p1)          # 128
        p2 = self.pool2(e2)
        p2 = self.noise2(p2)

        e3 = self.encoder3(p2)          # 256
        p3 = self.pool3(e3)
        p3 = self.noise3(p3)

        e4 = self.encoder4(p3)          # 512
        p4 = self.pool4(e4)
        p4 = self.noise4(p4)


        b = self.bottom(p4)             # 1024


        up4 = self.up4(b)               
        # Attention gate
        g_e4 = self.att4(g=up4, x=e4)
        cat4 = torch.cat([up4, g_e4], dim=1)  
        d4 = self.decoder4(cat4)       # 256


        up3 = self.up3(d4)             # to 256
        g_e3 = self.att3(g=up3, x=e3)
        cat3 = torch.cat([up3, g_e3], dim=1) 
        d3 = self.decoder3(cat3)       # 128

        up2 = self.up2(d3)             # to 128
        g_e2 = self.att2(g=up2, x=e2)
        cat2 = torch.cat([up2, g_e2], dim=1)  
        d2 = self.decoder2(cat2)       # 64

        up1 = self.up1(d2)             # to 64
        g_e1 = self.att1(g=up1, x=e1)
        cat1 = torch.cat([up1, g_e1], dim=1)  
        d1 = self.decoder1(cat1)       # 64

        out = self.out_conv(d1)
        return out

if __name__ == "__main__":
    model = UNetAttention()
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print("input:", x.shape, "output:", y.shape)
    print("params:", sum(p.numel() for p in model.parameters()))
