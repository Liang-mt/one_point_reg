import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import KeypointDataset,MyDataset
from net import KeypointDetector_v2_heatmap
import os

# 配置参数
save_path = 'train_heatmaps'
os.makedirs(save_path, exist_ok=True)

train_data_path=r'./datasets/data_center_train.txt'
val_data_path=r'./datasets/data_center_val.txt'
weight = "heatmap_model500.pth"

def train_model(json_dir, num_keypoints, img_size=128, num_epochs=500):
    # 创建数据集
    train_loader = DataLoader(MyDataset(train_data_path), batch_size=16, shuffle=True)
    test_loader = DataLoader(MyDataset(val_data_path), batch_size=16, shuffle=True)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointDetector_v2_heatmap(num_keypoints=num_keypoints).to(device)

    if os.path.exists(weight):
        model.load_state_dict(torch.load(weight))
        print('successful load weight！')
    else:
        print('not successful load weight')

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, heatmaps) in enumerate(train_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            # 前向传播
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, heatmaps)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可视化保存
            if batch_idx % 1 == 0:
                _image = images[0].cpu()
                _segment_image = heatmaps[0].cpu()
                _out_image = pred_heatmaps[0].cpu()
                img = torch.stack([_segment_image[0], _out_image[0]], dim=0)
                img = torch.unsqueeze(img, dim=1)
                save_image(img, f'{save_path}/{batch_idx}.png', nrow=2)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss}')

    if (epoch + 1) % 100 == 0:
        weight_path = f'./weights/heatmap_model{epoch + 1}.pth'  # 构建正确的文件路径
        torch.save(model.state_dict(), weight_path)  # 保存模型
        print("模型已保存")

if __name__ == "__main__":
    # 配置参数
    json_dir = "./data/"
    num_keypoints = 1  # 根据实际关键点数量设置
    img_size = 128  # 与数据集配置一致

    # 开始训练
    train_model(json_dir, num_keypoints, img_size)