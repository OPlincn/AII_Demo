import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from modelDefine import FashionMnistNet
import numpy as np
# 假设FashionMnistNet和其他必要的imports已经完成

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMnistNet().to(device)
model.load_state_dict(torch.load(rf'/Volumes/Extreme SSD/AITheory_FInal/save_modules/fashion_mnist_Mix32UP10no.pth', map_location=device))
model.eval()

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载测试数据集
test_set = FashionMNIST(root=rf'/Volumes/Extreme SSD/AITheory_FInal/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 预测测试集并获取置信度
def get_confidence(model, test_loader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidence, _ = torch.max(probabilities, 1)
            confidences.extend(confidence.cpu().numpy())
    return confidences

confidences = get_confidence(model, test_loader)

# 估计置信度的概率密度函数
density = gaussian_kde(confidences)
xs = np.linspace(0, 1, 200)
density.covariance_factor = lambda: .25
density._compute_covariance()

# 绘制置信度的概率密度函数
plt.figure(figsize=(8, 6))
plt.plot(xs, density(xs), color='red')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.title('Probability Density Function of Model Confidences')

# 计算并标注平均置信度
mean_confidence = np.mean(confidences)
plt.axvline(mean_confidence, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(mean_confidence*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(mean_confidence))

# 计算并标注中位数置信度
median_confidence = np.median(confidences)
plt.axvline(median_confidence, color='g', linestyle='dotted', linewidth=1)
plt.text(median_confidence*1.05, max_ylim*0.8, 'Median: {:.2f}'.format(median_confidence))

# 可以选择是否标注置信度大于0.9的区域
high_confidence_threshold = 0.9
plt.fill_betweenx([0, max(density(xs))], high_confidence_threshold, 1, color='gray', alpha=0.5)
plt.text(1, max_ylim*0.1, '>{:.1f} region'.format(high_confidence_threshold), ha='right')

plt.show()

