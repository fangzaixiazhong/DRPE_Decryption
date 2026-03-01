import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from unet import UNetDeep
from unet_attention import UNetAttention
import numpy as np
import torch.serialization



import torch
import numpy as np
torch.serialization.add_safe_globals([np.ndarray])

class DRPEMatcher: 
    def __init__(self, model_path): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = UNetAttention().to(self.device) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 
        self.model.eval() 
        self.transform = transforms.Compose([ transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ]) 
        print(f"模型加载成功: {model_path}")
    
    def ssim_loss(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)
        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return torch.clamp((1 - ssim_map.mean()) / 2, 0, 1)
    
    def decrypt_image(self, encrypted_image_path):

        encrypted_img = Image.open(encrypted_image_path).convert("L")
        original_size = encrypted_img.size
        
        input_tensor = self.transform(encrypted_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # 反归一化并转换为PIL图像
        output_tensor = (output_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        decrypted_img = Image.fromarray((output_tensor.numpy() * 255).astype(np.uint8))
        
        # 恢复到原始尺寸
        decrypted_img = decrypted_img.resize(original_size, Image.LANCZOS)
        
        return decrypted_img
    
    def compare_images_consistent(self, img1, img2, alpha=0.8):
  
        # 统一尺寸和预处理（与训练时相同）
        img1_processed = self.transform(img1).unsqueeze(0)
        img2_processed = self.transform(img2).unsqueeze(0)
        
        # L1 
        l1_distance = F.l1_loss(img1_processed, img2_processed).item()
        l1_similarity = 1 - l1_distance  # 转换为相似度
        
        # SSIM 
        ssim_distance = self.ssim_loss(img1_processed, img2_processed).item()
        ssim_similarity = 1 - ssim_distance  # 转换为相似度
        
        # 混合
        mixed_similarity = alpha * l1_similarity + (1 - alpha) * ssim_similarity
        
        mixed_similarity = max(0, min(1, mixed_similarity))
        
        return mixed_similarity
    
    def find_most_similar(self, decrypted_img, search_folder, top_k=1, alpha=0.6):

        if not os.path.exists(search_folder):
            raise FileNotFoundError(f"搜索文件夹不存在: {search_folder}")
        
        similarities = []
        
        for file in os.listdir(search_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                original_path = os.path.join(search_folder, file)
                
                try:
                    original_img = Image.open(original_path).convert("L")
                    

                    similarity = self.compare_images_consistent(decrypted_img, original_img, alpha=alpha)
                    
                    similarities.append({
                        'path': original_path,
                        'filename': file,
                        'similarity': similarity,
                        'l1_sim': None,
                        'ssim_sim': None
                    })
                except Exception as e:
                    print(f"⚠️ 处理图片 {file} 时出错: {e}")
                    continue
        
        # 按相似度排序（从高到低）
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]

def batch_process_encrypted_images_consistent(model_path, encrypted_folder, original_folder, save_folder=None, alpha=0.6):

    
    # 初始化匹配器
    matcher = DRPEMatcher(model_path)
    
    # 确保保存文件夹存在
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 获取所有加密图像文件
    encrypted_files = []
    for file in os.listdir(encrypted_folder):
        if file.endswith('_mag.png'):
            encrypted_files.append(file)
    
    print(f" 找到 {len(encrypted_files)} 个加密图像文件")
    print(f" 使用相似度度量: L1+SSIM混合 (alpha={alpha})")
    
    # 统计结果
    results = []
    correct_count = 0
    
    for encrypted_file in encrypted_files:
        try:

            encrypted_path = os.path.join(encrypted_folder, encrypted_file)
            
            # 预期的原图文件名
            expected_original = encrypted_file.replace('_mag.png', '')
            if expected_original.endswith('.jpg'):
                expected_original = expected_original
            else:
                expected_original = expected_original + '.jpg'
            
            print(f"\n处理: {encrypted_file}")
            print(f"期望匹配: {expected_original}")
            

            decrypted_img = matcher.decrypt_image(encrypted_path)

            if save_folder:
                decrypted_save_path = os.path.join(save_folder, f"decrypted_{expected_original}")
                decrypted_img.save(decrypted_save_path)
                print(f"解密图保存: {decrypted_save_path}")

            search_results = matcher.find_most_similar(decrypted_img, original_folder, top_k=1, alpha=alpha)
            
            if search_results:
                best_match = search_results[0]
                is_correct = (best_match['filename'] == expected_original)
                
                if is_correct:
                    correct_count += 1
                    status = "正确"
                else:
                    status = "错误"
                
                result = {
                    'encrypted_file': encrypted_file,
                    'expected_original': expected_original,
                    'matched_original': best_match['filename'],
                    'similarity': best_match['similarity'],
                    'is_correct': is_correct,
                    'status': status
                }
                results.append(result)
                
                print(f"{status} | 匹配到: {best_match['filename']} | 混合相似度: {best_match['similarity']:.4f}")
            else:
                print(f" 未找到匹配的原图")
                results.append({
                    'encrypted_file': encrypted_file,
                    'expected_original': expected_original,
                    'matched_original': None,
                    'similarity': 0,
                    'is_correct': False,
                    'status': ' 无匹配'
                })
                
        except Exception as e:
            print(f"处理 {encrypted_file} 时出错: {e}")
            results.append({
                'encrypted_file': encrypted_file,
                'expected_original': expected_original,
                'matched_original': None,
                'similarity': 0,
                'is_correct': False,
                'status': f' 错误: {str(e)}'
            })
    
    # 计算准确率
    accuracy = correct_count / len(encrypted_files) if encrypted_files else 0
    

    print(f"\n{'='*60}")
    print("批量处理结果总结（使用训练一致的相似度）")
    print(f"{'='*60}")
    print(f"总处理数量: {len(encrypted_files)}")
    print(f"正确匹配: {correct_count}")
    print(f"错误匹配: {len(encrypted_files) - correct_count}")
    print(f"准确率: {accuracy:.2%}")
    print(f"相似度度量: L1+SSIM混合 (alpha={alpha})")
    print(f"{'='*60}")
    
    return results, accuracy

def main():
    # 参数设置
    model_path = r""
    encrypted_image_path = r""
    search_folder = r""
    save_decrypted_path = r""
    
    # 使用与阶段2训练相同的权重
    alpha = 0.65
    
    try:
        print("开始批量处理加密图像...")
        results, accuracy = batch_process_encrypted_images_consistent(
            model_path=model_path,
            encrypted_folder=encrypted_image_path,
            original_folder=search_folder,
            save_folder=save_decrypted_path,
            alpha=alpha
        )
        
        print(f"\n 处理完成！最终准确率: {accuracy:.2%}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
