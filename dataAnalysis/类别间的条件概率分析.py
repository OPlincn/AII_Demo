from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from modelDefine import FashionMnistNet
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import seaborn as sns

# 假设你已经有了模型的输出和实际标签
# outputs = model(images) # 模型输出
# labels = ... # 实际标签
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformsTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])
test_set = torchvision.datasets.FashionMNIST(root='/Volumes/Extreme SSD/AITheory_FInal/data', train=False, download=True, transform=transformsTest)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True)
model = FashionMnistNet().to(device)
model.load_state_dict(torch.load('/Volumes/Extreme SSD/AITheory_FInal/save_modules/fashion_mnist_Mix32UP10no.pth', map_location='cpu'))
model.eval()

# 获取测试数据
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 进行预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)


# 计算混淆矩阵并归一化
conf_matrix = np.zeros((10, 10), dtype=int)
classes= ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 遍历测试集中的所有数据
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # 进行预测
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 更新混淆矩阵
    conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(10))

# 计算归一化混淆矩阵
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 绘制归一化后的混淆矩阵
plt.figure(figsize=(10, 10))
sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Normalized Confusion Matrix by LinWeiTeng')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()