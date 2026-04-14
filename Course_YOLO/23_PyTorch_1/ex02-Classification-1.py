import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from modle_CIFAR10 import CIFAR10

# 超参数
learning_rate = 0.01

# 1. 准备数据集
train_data = torchvision.datasets.CIFAR10("./data_cifar10", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./data_cifar10", train=False, download=True,
                                         transform = torchvision.transforms.ToTensor())

# 2. 训练数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# print(f"训练数据集的长度为：{train_data_size}")
# print("测试数据集的长度为：{}".format(test_data_size))

# 3. 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 4. 搭建神经网络模型
model = CIFAR10()
# print(model)

# 5. 创建模型的损失函数和优化器
# 损失函数
# loss_fn = CrossEntropyLoss(reduction='none')
loss_fn = CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 6. 设置训练网络的参数与纪录
epoch = 10
total_train_step = 0
total_test_step = 0

# 7. 撰写训练步骤，开始训练
for i in range(epoch):
    print("------ 第 {} 轮训练开始 ------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        outputs = model(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        # print(loss.item())

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss))

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的 Loss：{}".format(total_test_loss))
    print("整体测试集上的 Accuracy：{}".format(total_accuracy/test_data_size))
    total_test_step = total_test_step + 1