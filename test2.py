import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from version1 import ImprovedCNNNet

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNNNet().to(device)
model.load_state_dict(torch.load('./save_modules/fashion_mnist_cnn.pth', map_location="cpu"))
model.eval()

# 加载测试数据集
test_set = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=6, shuffle=True)

# 获取一些测试图像
images, labels = next(iter(test_loader))

images, labels = images.to(device), labels.to(device)

# 进行预测
with torch.no_grad():
    outputs = model(images)

# 获取预测类别
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().numpy()

# 类别标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 创建带有标签的图像网格
fig = plt.figure(figsize=(12, 6))
for idx in range(6):
    ax = fig.add_subplot(2, 3, idx+1, xticks=[], yticks=[])
    matplotlib_imshow(images[idx].cpu())
    ax.set_title(f'Actual: {class_names[labels[idx]]}\nPredicted: {class_names[predicted[idx]]}', loc='center')

# 保存图像到TensorBoard
writer = SummaryWriter('runs/fashion_mnist_test')
writer.add_figure('test_images_with_labels', fig)

writer.close()
