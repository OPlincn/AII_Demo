import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载Fashion MNIST数据集
train_set = torchvision.datasets.FashionMNIST(root=rf'/Volumes/Extreme SSD/AITheory_FInal/data', train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
images, _ = next(iter(train_loader))
images = images.numpy()

# 将图像数据重塑为二维数组（每行是一个展平的图像）
flattened_images = images.reshape(images.shape[0], -1)

# # 计算像素间的相关系数矩阵
# correlation_matrix = np.corrcoef(flattened_images, rowvar=False)

# # 绘制相关系数矩阵的热力图
# plt.figure(figsize=(15, 15))
# sns.heatmap(correlation_matrix, cmap='coolwarm')
# plt.title('Pixel Correlation Matrix')
# plt.show()

# 计算特征的协方差矩阵
covariance_matrix = np.cov(flattened_images, rowvar=False)

# 绘制协方差矩阵的热力图
plt.figure(figsize=(15, 15))
sns.heatmap(covariance_matrix, cmap='viridis')
plt.title('Feature Covariance Matrix by LinWeiTeng')
plt.show()

