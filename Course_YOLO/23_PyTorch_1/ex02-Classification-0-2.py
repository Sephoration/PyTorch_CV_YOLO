import time
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from modle_CIFAR10 import CIFAR10

# 超参数
learning_rate = 0.01
epoch = 10

writer = SummaryWriter("./logs")

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
# model = CIFAR10()
model = torch.load("steven_method1_1.pth", weights_only=False)

if torch.cuda.is_available():
    model = model.cuda()
# print(model)

# add_graph
# dummy_input = torch.randn(1, 3, 32, 32).cuda()
# writer.add_graph(model, dummy_input)

# 5. 创建模型的损失函数和优化器
# 损失函数
loss_fn = CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 6. 设置训练网络的参数与纪录
total_train_step = 0
total_test_step = 0

# 7. 撰写训练步骤，开始训练
for i in range(epoch):
    print("------ 第 {} 轮训练开始 ------".format(i+1))
    start_time = time.time()

    # 训练步骤开始
    for data in train_dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        # print(loss.item())

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            if i == 0:
                img_grid = torchvision.utils.make_grid(imgs[:32], nrow=8, normalize=True)
                writer.add_image("train_images", img_tensor=img_grid, global_step=total_train_step)

    end_time = time.time()
    print("第 {} 轮训练时间：{}".format(i+1, end_time-start_time))

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的 Loss：{}".format(total_test_loss))
    print("Test_Accuracy:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, i+1)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, i+1)

torch.save(model, "steven_method1_2.pth")
writer.close()
