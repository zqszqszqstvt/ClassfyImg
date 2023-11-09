
# 在测试集中取出一个批次的数据做图像和标签的展示
import torchvision.utils
from data import imgshow
from data import Net
import torchvision.transforms as transforms
import torch
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # # 打印原始图片
    # imgshow(torchvision.utils.make_grid(images))
    # # 打印真实的标签
    # print('真实标签:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    PATH = './cifar_net.pth'
    # 加载模型参数
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # # 预测
    # outputs = net(images)
    # # 选择概率最大的作为预测结果
    # _, predicted = torch.max(outputs, 1)
    # # 打印预测标签
    # print('预测：', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # 在测试集上测试准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('模型准确率：%d %%' % (100 * correct / total))
