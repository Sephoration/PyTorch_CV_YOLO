
# 6. 设置训练网络的参数与纪录

# 7. 撰写训练步骤，开始训练
for data in train_dataloader:
    optimizer.zero_grad()
    imgs, targets = data
    outputs = model(imgs)
    soft_outputs = torch.softmax(outputs, dim=1)  # 不该加这一步！
    loss = loss_fn(soft_outputs, targets)  # 错误❗
    # print(outputs.shape)
    loss = loss_fn(outputs, targets)

    loss.backward()
    optimizer.step()
    print(loss.item())




