import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 构建展示图片的函数
def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义线性层/全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 变化x的形状来适配全连接层的输入
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # # 从数据迭代器中读取一张图片
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # 展示图片
    # imgshow(torchvision.utils.make_grid(images))
    # # 打印标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    net = Net()
    # print(net)

    # 选用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 训练
    for epoch in range(2):
        running_loss = 0.0
        # 按批次迭代训练模型
        for i, data in enumerate(trainloader, 0):
            # 从data中取出含有输入图像的张量inputs,标签张量
            inputs, labels = data
            # 第一步梯度清零
            optimizer.zero_grad()
            # 第二步将图像输入进网络得到输出张量
            outputs = net(inputs)

            # 计算损失值
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            # 打印训练信息
            running_loss += loss.item()
            if(i + 1) % 2000 ==0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('训练结束！')
    # 保存模型
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)






