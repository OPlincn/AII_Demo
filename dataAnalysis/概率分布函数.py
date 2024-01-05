import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from modelDefine import FashionMnistNet
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
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
# Estimate the probability density function of confidences
density = gaussian_kde(confidences)
xs = np.linspace(0, 1, 200)
ys = density(xs)

# Calculate the cumulative distribution function (CDF) directly from the PDF
# Note: If the PDF is already normalized, we don't need to normalize it again.
cdf = np.cumsum(ys) * (xs[1] - xs[0])
cdf /= cdf[-1]  # Ensure the CDF goes from 0 to 1

# Calculate mean and median confidence
mean_confidence = np.mean(confidences)
median_confidence = np.median(confidences)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Plot PDF
axs[0].plot(xs, ys, color='red', label='PDF')
axs[0].set_xlabel('Confidence')
axs[0].set_ylabel('Density')
axs[0].set_title('Probability Density Function of Model Confidences by LinWeiTeng')
axs[0].axvline(mean_confidence, color='k', linestyle='dashed', linewidth=1)
axs[0].text(mean_confidence, max(ys)*0.95, f'Mean: {mean_confidence:.2f}', horizontalalignment='right')
axs[0].axvline(median_confidence, color='g', linestyle='dotted', linewidth=1)
# axs[0].text(median_confidence, max(ys)*0.85, f'Median: {median_confidence:.2f}', horizontalalignment='right')
axs[0].legend()

# Plot CDF
axs[1].plot(xs, cdf, color='blue', label='CDF')
axs[1].set_xlabel('Confidence')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_title('Cumulative Distribution Function of Model Confidences by LinWeiTeng')
axs[1].axhline(0.5, color='gray', linestyle='dashed', linewidth=1)
axs[1].axvline(median_confidence, color='g', linestyle='dotted', linewidth=1)
# axs[1].text(median_confidence, 0.5, f'Median: {median_confidence:.2f}', verticalalignment='bottom', horizontalalignment='right')
axs[1].legend()

plt.tight_layout()
plt.show()