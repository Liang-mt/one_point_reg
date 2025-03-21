import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from net import KeypointDetector,KeypointDetector_v1,KeypointDetector_v2
from datasets import KeypointDataset
from torchvision.utils import save_image

save_path='train_image'

def train_model(json_dir, num_keypoints, num_epochs=300):
    # 创建数据集
    dataset = KeypointDataset(json_dir)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 根据关键点数量创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointDetector(num_keypoints).to(device)

    # 使用Smooth L1 Loss
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'{epoch + 1}-{i}-train_loss===>>{loss.item()}')

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

    # 保存模型
    torch.save(model.state_dict(), "./weights/custom_keypoint_model_V2.pth")



if __name__ == "__main__":
    # 训练配置
    json_dir = "./data/"
    num_keypoints = 1  # 根据实际关键点数量设置

    # 训练模型
    train_model(json_dir, num_keypoints)