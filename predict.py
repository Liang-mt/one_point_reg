# import os
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# from net import KeypointDetector,KeypointDetector_v1,KeypointDetector_v2
#
# class AdvancedKeypointPredictor:
#     def __init__(self, model_path, num_keypoints, img_size=128):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = KeypointDetector_v2(num_keypoints).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=False))
#         self.model.eval()
#
#         self.transform = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         self.img_size = img_size
#
#     def predict(self, img_path):
#         # 加载原始图像
#         image = Image.open(img_path).convert("RGB")
#         orig_w, orig_h = image.size
#
#         # 预处理
#         image_tensor = self.transform(image).unsqueeze(0).to(self.device)
#
#         # 预测
#         with torch.no_grad():
#             outputs = self.model(image_tensor)
#
#         # 后处理
#         keypoints = outputs.cpu().numpy().reshape(-1, 2)
#
#         # 还原到原始尺寸
#         keypoints[:, 0] *= orig_w
#         keypoints[:, 1] *= orig_h
#
#         return keypoints.astype(int)  # 转换为整数坐标
#
#     def visualize_and_save(self, img_path, keypoints, output_dir, connections=None):
#         """
#         使用OpenCV绘制并保存结果
#         :param img_path: 输入图片路径
#         :param keypoints: 关键点坐标数组
#         :param output_dir: 输出目录
#         :param connections: 连接关系列表，例如 [(0,1), (1,2)]。如果为None，则自动判断是否需要连接关系。
#         """
#         # 创建输出目录
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 用PIL读取图片并转换为OpenCV格式
#         image = Image.open(img_path).convert("RGB")
#         img_np = np.array(image)
#
#         # 转换颜色通道 (RGB -> BGR)
#         img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#
#         # 自动判断是否需要连接关系
#         if connections is None:
#             if len(keypoints) > 1:  # 如果关键点数量大于1，则自动生成连接关系
#                 connections = [(i, i + 1) for i in range(len(keypoints) - 1)]
#             else:  # 如果关键点数量为1，则不绘制连接关系
#                 connections = []
#
#         # 绘制连接线
#         for (start, end) in connections:
#             x1, y1 = keypoints[start]
#             x2, y2 = keypoints[end]
#             cv2.line(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色线条
#
#         # 绘制关键点
#         for (x, y) in keypoints:
#             cv2.circle(img_cv, (x, y), 1, (0, 0, 255), -1)  # 红色实心圆
#
#         # 生成保存路径
#         filename = os.path.basename(img_path)
#         save_path = os.path.join(output_dir, f"kp_{filename}")
#
#         # 保存图片
#         cv2.imwrite(save_path, img_cv)
#         print(f"结果已保存至：{save_path}")
#         return save_path
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化预测器
#     num_keypoints = 1  # 根据实际关键点数量设置
#     predictor = AdvancedKeypointPredictor("custom_keypoint_model_V2.pth", num_keypoints)
#
#     # 测试图片路径
#     test_img = "./data/579.png"
#
#     # 预测关键点
#     predicted_keypoints = predictor.predict(test_img)
#
#     # 可视化并保存结果
#     output_directory = "results"  # 指定输出目录
#     predictor.visualize_and_save(
#         img_path=test_img,
#         keypoints=predicted_keypoints,
#         output_dir=output_directory,
#         connections=None  # 自动判断是否需要连接关系
#     )


import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from net import KeypointDetector, KeypointDetector_v1, KeypointDetector_v2


class AdvancedKeypointPredictor:
    def __init__(self, model_path, num_keypoints, img_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KeypointDetector(num_keypoints).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_size = img_size

    def predict(self, img_path):
        """预测单张图片"""
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.cpu().numpy().reshape(-1, 2)
        keypoints[:, 0] *= orig_w
        keypoints[:, 1] *= orig_h
        return keypoints.astype(int)

    def process_folder(self, input_dir, output_dir, connections=None):
        """
        批量处理文件夹中的图片
        :param input_dir: 输入图片文件夹路径
        :param output_dir: 输出结果文件夹路径
        :param connections: 关键点连接关系
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        if not image_files:
            print(f"未在 {input_dir} 中找到图片文件")
            return

        print(f"开始处理 {len(image_files)} 张图片...")

        for idx, filename in enumerate(image_files, 1):
            try:
                # 处理单张图片
                img_path = os.path.join(input_dir, filename)
                keypoints = self.predict(img_path)

                # 可视化并保存
                self._visualize_and_save(
                    img_path=img_path,
                    keypoints=keypoints,
                    output_dir=output_dir,
                    connections=connections
                )

                print(f"处理进度: {idx}/{len(image_files)} - {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

    def _visualize_and_save(self, img_path, keypoints, output_dir, connections=None):
        """内部可视化方法"""
        # 读取图片并转换颜色空间
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 自动生成连接关系
        if connections is None and len(keypoints) > 1:
            connections = [(i, i + 1) for i in range(len(keypoints) - 1)]

        # 绘制连接线
        if connections:
            for (start, end) in connections:
                x1, y1 = keypoints[start]
                x2, y2 = keypoints[end]
                cv2.line(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制关键点（放大显示）
        for (x, y) in keypoints:
            cv2.circle(img_cv, (x, y), 1, (0, 0, 255), -1)  # 红色实心圆

        # 保存结果
        output_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img_cv)


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = AdvancedKeypointPredictor(
        model_path="custom_keypoint_model.pth",
        num_keypoints=1  # 根据实际关键点数量设置
    )

    # 设置输入输出路径
    input_folder = "./test_image"  # 原始图片文件夹
    output_folder = "./test_result"  # 处理结果文件夹

    # 批量处理整个文件夹
    predictor.process_folder(
        input_dir=input_folder,
        output_dir=output_folder,
        connections=None  # 自动判断连接关系
    )