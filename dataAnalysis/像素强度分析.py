import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置数据转换 - 将PIL图像转换为张量
transform = transforms.Compose([transforms.ToTensor()])

# 加载Fashion MNIST数据集，并应用转换
train_set = torchvision.datasets.FashionMNIST(root=rf'/Volumes/Extreme SSD/AITheory_FInal/data', train=True, download=True, transform=transform)

# 使用DataLoader加载数据集
train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)

# 获取所有训练图像
train_images, _ = next(iter(train_loader))
train_images = train_images.numpy()

# 计算像素强度
pixel_intensities = train_images.flatten()

# 绘制直方图
plt.figure(figsize=(10, 5))
plt.hist(pixel_intensities, bins=50, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution in Fashion MNIST by LinWeiTeng')
plt.show()
