import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Squeeze-Excitation模块，用于增强特征通道间的依赖关系
class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        # 自适应平均池化层，输出大小为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化层，输出大小为1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 定义全连接层，用于特征压缩和激活
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            Swish(),  # 使用Swish激活函数
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 平均池化和最大池化
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        # 拼接池化后的特征
        y = torch.cat((y_avg, y_max), dim=1)
        # 通过全连接层并调整形状
        y = self.fc(y).view(b, c, 1, 1)
        # 与原始输入相乘，实现特征重标定
        return x * y.expand_as(x)

# 残差块，用于构建深度残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，保持通道数不变
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 第二个卷积层，使用膨胀卷积保持尺寸不变
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.3)
        # Squeeze-Excitation模块
        self.se = SqueezeExcitation(in_channels)
        
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual  # 残差连接
        out = F.relu(out)
        return out


# Improved CNN Model
class FashionMnistNet(nn.Module):
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU()
        self.resblock0 = ResidualBlock(32)
        self.dropout0 = nn.Dropout(0.3)
        
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.resblock1 = ResidualBlock(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.resblock2 = ResidualBlock(128)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.resblock3 = ResidualBlock(256)
        self.dropout3 = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.resblock0(x)
        x = self.dropout0(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.dropout2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.resblock3(x)
        x = self.dropout3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



# 加载你的模型（确保你的模型已经训练完成）
# model = ... # 这里应该是你加载已经训练好的模型的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMnistNet().to(device)
model.load_state_dict(torch.load(rf'/Volumes/Extreme SSD/AITheory_FInal/save_modules/fashion_mnist_Mix32UP10no.pth', map_location="cpu"))
model.eval()
# 数据预处理

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载测试数据集
test_set = FashionMNIST(root=rf'/Volumes/Extreme SSD/AITheory_FInal/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# 测试模型并计算准确率
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

# 绘制置信度分布
plt.figure(figsize=(6, 4))
plt.hist(confidences, bins=20, color='blue', alpha=0.7)
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution of Model Predictions by LinWeiTeng')
plt.show()
