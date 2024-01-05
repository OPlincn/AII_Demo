import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from version2 import ImprovedCNNNet
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNNNet().to(device)
model.load_state_dict(torch.load('./save_modules/fashion_mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# 加载测试数据集
test_set = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=12, shuffle=False)

# 获取一些测试图像
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 进行预测
with torch.no_grad():
    outputs = model(images)

# 获取预测类别
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().numpy()

# 可视化图像和标签
writer = SummaryWriter('./runs/fashion_mnist_test')
img_grid = make_grid(images.cpu())
writer.add_image('six_fashion_mnist_images', img_grid)

# 将标签添加到TensorBoard日志
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predicted_labels = [class_names[i] for i in predicted]
writer.add_text('predicted_labels', ', '.join(predicted_labels))
print(predicted_labels)

writer.close()
