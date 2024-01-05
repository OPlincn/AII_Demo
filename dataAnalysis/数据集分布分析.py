import matplotlib.pyplot as plt
import torchvision
from collections import Counter

# 加载Fashion MNIST数据集
train_set = torchvision.datasets.FashionMNIST(root='/Volumes/Extreme SSD/AITheory_FInal/data', train=True, download=True)
test_set = torchvision.datasets.FashionMNIST(root='/Volumes/Extreme SSD/AITheory_FInal/data', train=False, download=True)

# 统计训练集和测试集中各类别的数量
train_counter = Counter(train_set.targets.numpy())
test_counter = Counter(test_set.targets.numpy())

# 类别标签
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 绘制条形图
plt.figure(figsize=(10, 5))
plt.bar(classes, [train_counter[i] for i in range(10)], alpha=0.5, label='Train')
plt.bar(classes, [test_counter[i] for i in range(10)], alpha=0.5, label='Test')
plt.xlabel('Classes')
plt.ylabel('Number of samples')
plt.title('Number of Samples per Class by LinWeiTeng')
plt.legend()
plt.show()
