import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import BoundarySupervisionNet

# 模拟数据集 (实际使用时替换为CASIA或COVERAGE数据集路径)
class TamperDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        # 模拟输入图像 (3通道) 和 掩码 (1通道)
        image = np.random.rand(3, 256, 256).astype(np.float32)
        mask = np.random.randint(0, 2, (1, 256, 256)).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundarySupervisionNet().to(device)
    criterion = nn.BCELoss() # 二值交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TamperDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("开始训练...")
    for epoch in range(5): # 演示5个epoch
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # 获取双输出：区域图 和 边界图
            region_pred, boundary_pred = model(images)
            
            # 复合损失函数 (边界监督策略)
            loss_region = criterion(region_pred, masks)
            loss_boundary = criterion(boundary_pred, masks)
            loss = loss_region + 0.5 * loss_boundary # 综合损失
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "tamper_detection_model.pth")
    print("模型训练完成并保存。")

if __name__ == "__main__":
    main()
