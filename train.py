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




# 训练模型
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}"), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:    # 每1000个cmini-batches
            writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
            print(rf"training loss: {running_loss / 100:.3f} ")
            running_loss = 0.0

# 测试模型，并返回验证损失
bestAccuracy = 0.
isSave = False
def test(epoch):
    global bestAccuracy
    global isSave
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    if bestAccuracy < accuracy:
        bestAccuracy = accuracy
        isSave = True
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

    return test_loss


if __name__ == "__main__":
# 加载数据集
# 设定设备
    # Define the model
    model = FashionMnistNet()
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transformTrain = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),      # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    transformsTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformTrain)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformsTest)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
    model = FashionMnistNet().to(device)
    # 损失函数和AdamW优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)

    # TensorBoard
    writer = SummaryWriter('runs/fashion_mnist_experiment')
    totalEpoch = 175
    # 执行训练和测试
    for epoch in range(totalEpoch):
        train(epoch)
        test_loss = test(epoch)
        scheduler.step(test_loss)
        if epoch >= totalEpoch - 15:
            train_set.transform=transformsTest
        if isSave:
            isSave = False
            torch.save(model.state_dict(), './save_modules/fashion_mnist_Mix32UP175.pth')
            

    writer.close()
