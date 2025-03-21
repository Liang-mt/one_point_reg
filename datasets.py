import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from heatmap_lable import CenterLabelHeatMap

class KeypointDataset(Dataset):
    def __init__(self, json_dir, img_size=128, transform=None):
        self.json_dir = json_dir
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 自动扫描所有JSON文件并验证对应图片存在
        self.annotations = []
        self.img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']  # 支持的图片格式

        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json'):
                json_path = os.path.join(json_dir, json_file)
                with open(json_path) as f:
                    data = json.load(f)

                # 获取文件名主干（不带扩展名）
                base_name = os.path.splitext(json_file)[0]

                # 查找对应图片文件
                img_path = self.find_image_file(base_name)
                if img_path:
                    # 更新数据结构
                    data['base_name'] = base_name
                    data['valid_img_path'] = img_path
                    self.annotations.append(data)
                else:
                    print(f"Warning: No image found for {base_name}")

    def find_image_file(self, base_name):
        """根据基础文件名查找存在的图片文件"""
        for ext in self.img_extensions:
            candidate_path = os.path.join(self.json_dir, f"{base_name}{ext}")
            if os.path.exists(candidate_path):
                return candidate_path
        return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]

        # 加载图像（使用验证过的路径）
        image = Image.open(data['valid_img_path']).convert("RGB")

        # 获取原始尺寸
        orig_w = data['imageWidth']
        orig_h = data['imageHeight']

        # 提取并归一化关键点
        keypoints = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'point':
                x, y = shape['points'][0]
                # 归一化到[0,1]
                x_norm = x / orig_w
                y_norm = y / orig_h
                keypoints.extend([x_norm, y_norm])

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(keypoints, dtype=torch.float32)


from torchvision import transforms
from heatmap_lable import *
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        f=open(path)
        self.dataset=f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data=self.dataset[index]
        img_path=data.split(' ')[0]
        image=Image.open(img_path).convert('RGB').resize((128,128))
        points=data.split(' ')[1:]
        points=[int(i.rstrip("\n"))*1.0 for i in points]
        label=[]
        for i in range(0,len(points),2):
            heatmap=CenterLabelHeatMap(128,128,points[i],points[i+1],5) #核的大小可以自己调整
            label.append(heatmap)
        label=np.stack(label)
        return transform(image),torch.Tensor(label)

if __name__ == '__main__':
    data=MyDataset('./datasets/data_center_val.txt')
    print(data[5][0].shape)
    print(data[5][1].shape)
    for i in data:
        print(i[0].shape)


