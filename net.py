import torch
import torch.nn as nn


class KeypointDetector(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 64, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 32, 32)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (256, 16, 16)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (512, 8, 8)
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为向量
            nn.Linear(512 * 8 * 8, 1024),  # 输入维度: 512*8*8, 输出维度: 1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_keypoints * 2)  # 输出维度: num_keypoints * 2
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        return self.regressor(features)  # 回归关键点



class KeypointDetector_v1(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v1, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络（减少通道数）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (256, 8, 8)
        )

        # 回归头（减少全连接层维度）
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为向量
            nn.Linear(256 * 8 * 8, 512),  # 输入维度: 256*8*8, 输出维度: 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_keypoints * 2)  # 输出维度: num_keypoints * 2
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        return self.regressor(features)  # 回归关键点

class KeypointDetector_v2(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v2, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (256, 8, 8)
        )

        # 全局平均池化 + 回归头
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出尺寸: (256, 1, 1)
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为向量
            nn.Linear(256, num_keypoints * 2)  # 输入维度: 256, 输出维度: num_keypoints * 2
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        pooled = self.global_pool(features)  # 全局平均池化
        return self.regressor(pooled)  # 回归关键点

class KeypointDetector_v2_heatmap(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v2_heatmap, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (256, 8, 8)
        )

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 输出尺寸: (128, 16, 16)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 输出尺寸: (64, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 输出尺寸: (32, 64, 64)
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, kernel_size=2, stride=2),  # 输出尺寸: (num_keypoints, 128, 128)
            nn.Sigmoid()  # 输出值在 [0, 1] 范围内
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        heatmaps = self.upsample(features)  # 上采样生成热力图
        return heatmaps






