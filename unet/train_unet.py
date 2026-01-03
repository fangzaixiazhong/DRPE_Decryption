import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from unet_attention import UNetAttention
from unet import UNetDeep
import numpy as np

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


class MixedLoss(nn.Module):
    def __init__(self, alpha=1):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_val = ssim_loss(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_val


class DRPEDataset(Dataset):
    def __init__(self, raw_dir, encrypted_dir, folders, split="train", transform=None):

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
    fp.write("")
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


    print("训练文件夹:", train_datasets)
    print("测试文件夹:", test_datasets)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
    model = UNetAttention().to(device)

    criterion = MixedLoss(alpha=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr = 1e-7)
    #scheduler = torch.optim.lr_scheduler.StepLR(
    #optimizer,
    #step_size=16,
    #gamma=0.5
#)


    best_loss = float("inf")
    num_epochs = 256
    import datetime
    start_time = datetime.datetime.now()
    fp.write(f"=== 训练开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    fp.write(f"训练文件夹: {train_datasets}\n")
    fp.write(f"测试文件夹: {test_datasets}\n")
    fp.write(f"设备: {device}\n")
    fp.write(f"总轮次: {num_epochs}\n")
    fp.write("=" * 50 + "\n")
    fp.flush()

    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()  # 记录每轮开始时间
        model.train()
        total_loss = 0

        for enc_img, raw_img in train_loader:
            enc_img, raw_img = enc_img.to(device), raw_img.to(device)

            # 🔹 随机加入轻微噪声
            if np.random.rand() < 0.22:
                noise = torch.randn_like(enc_img) * 0.01
                enc_img = torch.clamp(enc_img + noise, -1, 1)

            outputs = model(enc_img)
            loss = criterion(outputs, raw_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - 训练损失: {avg_loss:.6f}")


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for enc_img, raw_img in test_loader:
                enc_img, raw_img = enc_img.to(device), raw_img.to(device)
                outputs = model(enc_img)
                loss = criterion(outputs, raw_img)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"→ 测试损失: {avg_test_loss:.6f}\n")

        # 🔻 调整学习率
        scheduler.step(avg_test_loss)
        #scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

        # 记录本轮结果到日志文件
        epoch_end_time = datetime.datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        log_line = (f"Epoch {epoch+1:02d} | "
                   f"时间: {epoch_end_time.strftime('%H:%M:%S')} | "
                   f"耗时: {epoch_duration:.1f}s | "
                   f"训练损失: {avg_loss:.6f} | "
                   f"测试损失: {avg_test_loss:.6f} | "
                   f"学习率: {current_lr:.2e}")
        
        fp.write(log_line + "\n")

        # 保存最优模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            save_path = fr"./best_atten_unet_L1_n.pth"
            torch.save(model.state_dict(), save_path)
            print(f"最优模型已更新并保存: {save_path}")
            fp.write(f"Epoch {epoch+1}: 最优模型已保存 (测试损失: {avg_test_loss:.6f})\n")

        # 每轮备份一次
        if epoch%5==0:
            torch.save(model.state_dict(), fr"./ssim_epoch_{epoch+1}.pth")
        

        fp.flush()

    # 记录训练结束信息
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟
    fp.write("=" * 50 + "\n")
    fp.write(f"=== 训练结束于: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    fp.write(f"总耗时: {total_duration:.1f} 分钟\n")
    fp.write(f"最佳测试损失: {best_loss:.6f}\n")
    fp.write("=" * 50 + "\n\n")
    
    # 关闭日志文件
    fp.close()
    
    print(f"训练完成！最佳测试损失: {best_loss:.6f}")
    print(f"详细日志已保存")