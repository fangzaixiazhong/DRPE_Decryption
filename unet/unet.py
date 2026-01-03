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

class UNetDeep(nn.Module):
    def __init__(self):
        super(UNetDeep, self).__init__()

        # 特征噪声
        self.noise1 = FeatureNoise(sigma=0.02)
        self.noise2 = FeatureNoise(sigma=0.015) 
        self.noise3 = FeatureNoise(sigma=0.01)
        self.noise4 = FeatureNoise(sigma=0.005)

        self.encoder1 = nn.Sequential(
            CBR(1, 64, dropout_p=0.02),
            CBR(64, 64, dropout_p=0.02)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            CBR(64, 128, dropout_p=0.04),
            CBR(128, 128, dropout_p=0.04)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            CBR(128, 256, dropout_p=0.06),
            CBR(256, 256, dropout_p=0.06)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 4层编码器
        self.encoder4 = nn.Sequential(
            CBR(256, 512, dropout_p=0.08),
            CBR(512, 512, dropout_p=0.08)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        #button
        self.bottom = nn.Sequential(
            CBR(512, 1024, dropout_p=0.10),  
            CBR(1024, 1024, dropout_p=0.10)
        )


        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            CBR(1024, 512, dropout_p=0.08), 
            CBR(512, 256, dropout_p=0.08)
        )

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            CBR(512, 256, dropout_p=0.04), 
            CBR(256, 128, dropout_p=0.04)
        )

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            CBR(256, 128, dropout_p=0.02),  
            CBR(128, 64, dropout_p=0.02)
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            CBR(128, 64, dropout_p=0.01),  # up1(64) + e1(64) = 128
            CBR(64, 64, dropout_p=0.01)   
        )


        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  

    def forward(self, x):

        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        p1 = self.noise1(p1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        p2 = self.noise2(p2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        p3 = self.noise3(p3)


        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        p4 = self.noise4(p4)

        # 底部
        b = self.bottom(p4)

        # 解码

        up4 = self.up4(b)
        cat4 = torch.cat([up4, e4], dim=1)
        d4 = self.decoder4(cat4)

        up3 = self.up3(d4)
        cat3 = torch.cat([up3, e3], dim=1)
        d3 = self.decoder3(cat3)

        up2 = self.up2(d3)
        cat2 = torch.cat([up2, e2], dim=1)
        d2 = self.decoder2(cat2)

        up1 = self.up1(d2)
        cat1 = torch.cat([up1, e1], dim=1)
        d1 = self.decoder1(cat1)

        out = self.out_conv(d1)
        return out

# 测试
if __name__ == "__main__":
    model = UNetDeep()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 统计dropout层数量
    dropout_layers = sum(1 for m in model.modules() if isinstance(m, nn.Dropout2d))
    print(f"Dropout层数量: {dropout_layers}")
    
    # 输出各层通道信息
    print("\n各层通道信息:")
    print("编码器: 1 → 64 → 128 → 256 → 512")
    print("底部: 512 → 1024 → 1024") 
    print("解码器: 1024 → 512 → 256 → 128 → 64 → 1")
    print("最终输出通道: 64")
