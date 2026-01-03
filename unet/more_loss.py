# losses_combined.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
import math
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from unet_attention import UNetAttention
from torch.utils.data import Dataset
def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return torch.clamp((1 - ssim_map.mean()) / 2, 0, 1)
class EdgeDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)

        # 注册成 buffer，使其自动随 model.to(device) 移动
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def get_edges(self, img):
        # 自动使用 self.sobel_x 所在的 device
        gx = torch.nn.functional.conv2d(img, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred_raw, target_raw):
        pred = torch.sigmoid(pred_raw)
        target = (target_raw + 1) / 2

        pred_edge = self.get_edges(pred)
        target_edge = self.get_edges(target)

        intersection = (pred_edge * target_edge).sum(dim=(1, 2, 3))
        union = pred_edge.sum(dim=(1, 2, 3)) + target_edge.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()



class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        return self.weight * (dx + dy)

class EdgeLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        # pred/target shape [B,1,H,W] in same device
        px = F.conv2d(pred, self.sobel_x, padding=1)
        py = F.conv2d(pred, self.sobel_y, padding=1)
        tx = F.conv2d(target, self.sobel_x, padding=1)
        ty = F.conv2d(target, self.sobel_y, padding=1)
        return self.weight * (F.l1_loss(px, tx) + F.l1_loss(py, ty)) / 2.0

class GSLoss(nn.Module):
    """Gradient Similarity Loss"""
    def __init__(self, weight=1.0, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred, target):
        # 计算简单梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :,  :, 1:] - pred[:, :, :-1, :]
        targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        targ_dy = target[:, :, :, 1:] - target[:, :, :-1, :]

        pred_mag = torch.sqrt(pred_dx[:, :, :, :-1]**2 + pred_dy[:, :, :-1, :]**2 + self.eps)
        targ_mag = torch.sqrt(targ_dx[:, :, :, :-1]**2 + targ_dy[:, :, :-1, :]**2 + self.eps)

        # avoid size mismatch: simply use absolute diff of grads (simpler and stable)
        loss_dx = F.l1_loss(pred_dx, targ_dx)
        loss_dy = F.l1_loss(pred_dy, targ_dy)
        return self.weight * 0.5 * (loss_dx + loss_dy)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, weight=1.0):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]  # 小片段
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device).eval()
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1))
        self.weight = weight

    def forward(self, pred, target):

        if pred.shape[1] == 1:
            pred3 = pred.repeat(1,3,1,1)
            targ3 = target.repeat(1,3,1,1)
        else:
            pred3 = pred
            targ3 = target
        pred_norm = (pred3 - self.mean.to(pred.device)) / self.std.to(pred.device)
        targ_norm = (targ3 - self.mean.to(target.device)) / self.std.to(target.device)
        f_pred = self.vgg(pred_norm)
        f_targ = self.vgg(targ_norm)
        return self.weight * F.l1_loss(f_pred, f_targ)


class CombinedLoss(nn.Module):
    def __init__(self, device,
                 w_l1=0.4, w_msssim=0.11, w_vgg=0.02, w_edge=0.23, w_tv=0.02, w_dice = 0.22):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_l1 = w_l1
        self.w_msssim = w_msssim
        self.w_vgg = w_vgg
        self.w_edge = w_edge
        self.w_tv = w_tv
        self.w_dice = w_dice
        self.device = device
        self.vgg = VGGPerceptualLoss(device, weight=1.0) if w_vgg > 0 else None
        self.edge = EdgeLoss(weight=1.0) if w_edge > 0 else None
        self.tv = TVLoss(weight=1.0) if w_tv > 0 else None
        self.dice = EdgeDiceLoss() if w_dice > 0 else None

    def forward(self, pred_raw, target_raw):
        # 先映射到 [0,1] 供 ms_ssim 和 vgg 使用
        pred = (pred_raw + 1.0) / 2.0
        target = (target_raw + 1.0) / 2.0

        l1_val = self.l1(pred_raw, target_raw)
        # ms_ssim: 输入范围 [0,1], 返回值相似度 [0,1] -> 损失 1 - ms_ssim
        msssim_val = 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True,win_size = 7,  weights=[0.5, 0.3, 0.2])

        loss = self.w_l1 * l1_val + self.w_msssim * msssim_val

        if self.w_vgg > 0 and self.vgg is not None:
            loss = loss + self.w_vgg * self.vgg(pred, target)
        if self.w_edge > 0 and self.edge is not None:
            loss = loss + self.w_edge * self.edge(pred_raw, target_raw)
        if self.w_tv > 0 and self.tv is not None:
            loss = loss + self.w_tv * self.tv(pred_raw)
        if self.w_dice > 0 and self.dice is not None:
            loss = loss + self.w_dice * self.dice(pred_raw, target_raw)

        return loss
class DRPEDataset(Dataset):
    def __init__(self, raw_dir, encrypted_dir, folders, split="train", transform=None):
        """
        split: 'train' or 'test'
        """
        self.image_pairs = []
        self.transform = transform

        assert split in ["train", "test"]
        self.split = split

        for folder in folders:
            raw_path = os.path.join(raw_dir, folder)
            enc_path = os.path.join(encrypted_dir, folder)
            if not os.path.exists(raw_path):
                continue

            files = sorted([
                f for f in os.listdir(raw_path)
                if f.endswith((".jpg", ".png"))
            ])

            if len(files) == 0:
                continue
            split_idx = int(0.8 * len(files))
            if self.split == "train":
                selected_files = files[:split_idx]
            else:
                selected_files = files[split_idx:]

            for file in selected_files:
                raw_img_path = os.path.join(raw_path, file)
                enc_img_name = file + "_mag.png"
                enc_img_path = os.path.join(enc_path, enc_img_name)

                if os.path.exists(enc_img_path):
                    self.image_pairs.append((enc_img_path, raw_img_path))

        print(f"{self.split} dataset initialized, total pairs: {len(self.image_pairs)}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        enc_path, raw_path = self.image_pairs[idx]

        enc_img = Image.open(enc_path).convert("L")
        raw_img = Image.open(raw_path).convert("L")

        if self.transform:
            enc_img = self.transform(enc_img)
            raw_img = self.transform(raw_img)

        return enc_img, raw_img
if __name__ == "__main__":
    base_dir = r""
    raw_dir = os.path.join(base_dir, "grey")
    encrypted_dir = os.path.join(base_dir, "drpe_encrypted")
    fp = open('','a')
    fp.write("最终打磨 lowdice\n")
    from torchvision import transforms

    transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),              # [0,255] → [0,1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # → [-1,1]
])
    all_folders = [
    f for f in os.listdir(raw_dir)
    if os.path.isdir(os.path.join(raw_dir, f))
]

    train_datasets = DRPEDataset(
    raw_dir=raw_dir,
    encrypted_dir=encrypted_dir,
    folders=all_folders,
    split="train",
    transform=transform
)

    test_datasets = DRPEDataset(
    raw_dir=raw_dir,
    encrypted_dir=encrypted_dir,
    folders=all_folders,
    split="test",
    transform=transform
)

    print(f"训练文件夹数: {len(train_datasets)}, 测试文件夹数: {len(test_datasets)}")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(
    train_datasets,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

    test_loader = DataLoader(
    test_datasets,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = UNetAttention().to(device)

    # 加载预训练模型,如果没有则从头开始训练
    checkpoint_path = r"./best_atten_unet_L1_al.pth"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict({k: v for k, v in ckpt.items() if k in model.state_dict()}, strict=False)
        print("加载预训练模型")
        fp.write("模型已加载")


    criterion = CombinedLoss(device=device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-7
    )
    best_val = 0.35
    num_epochs = 60

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0

        for enc_img, raw_img in train_loader:
            enc_img, raw_img = enc_img.to(device), raw_img.to(device)
            optimizer.zero_grad()
            pred = model(enc_img)
            loss = criterion(pred, raw_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_img, raw_img in test_loader:
                enc_img, raw_img = enc_img.to(device), raw_img.to(device)
                pred = model(enc_img)
                loss = criterion(pred, raw_img)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch}/{num_epochs}] train={train_loss:.6f}  val={val_loss:.6f}")
        fp.write(f"[{epoch}/{num_epochs}] train={train_loss:.6f}  val={val_loss:.6f}")
        fp.write(f" lr:{current_lr}\n")
        fp.flush()
        if val_loss < best_val:
            best_val = val_loss
            save_path = f"./improved_L1.pth"
            torch.save(model.state_dict(), save_path)
            print("  -> 新最佳模型已保存")
            fp.write("模型已保存\n")
        if epoch % 5 == 1:
            save_path = f"./detail_epoch{epoch}.pth"
            torch.save(model.state_dict(), save_path)
    fp.write("训练结束\n")
    fp.close()
