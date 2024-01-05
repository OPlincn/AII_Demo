import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from modelDefine import FashionMnistNet



# 加载你的模型（确保你的模型已经训练完成）
# model = ... # 这里应该是你加载已经训练好的模型的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMnistNet().to(device)
model.load_state_dict(torch.load(rf'/Volumes/Extreme SSD/AITheory_FInal/save_modules/fashion_mnist_Mix32UP10no.pth', map_location="cpu"))
model.eval()
# 数据预处理

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载测试数据集
test_set = FashionMNIST(root='/Volumes/Extreme SSD/AITheory_FInal/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# 测试模型并计算准确率
def get_test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

accuracy = get_test_accuracy(model, test_loader)
print(accuracy)

# 绘制准确率分布（在这个案例中，我们只有一个准确率值）
plt.figure(figsize=(6, 4))
plt.hist([accuracy], bins=10, color='green', alpha=0.7)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Model Accuracy Distribution on Test Set by LinWeiTeng')
plt.show()
