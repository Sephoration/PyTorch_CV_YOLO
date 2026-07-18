import matplotlib.pyplot as plt

# 数据：按 epoch 顺序
epochs = list(range(1, 11))
test_losses = [
    0.3724,
    0.2477,
    0.3699,
    0.2098,
    0.2744,
    0.1882,
    0.2450,
    0.2757,
    0.2061,
    0.2812
]
test_accuracies = [
    0.8863,
    0.9332,
    0.8770,
    0.9325,
    0.9104,
    0.9439,
    0.9183,
    0.9147,
    0.9389,
    0.9225
]

# 创建图形
plt.figure(figsize=(14, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, test_losses, marker='o', color='red', linewidth=2, markersize=6)
plt.title('Test Loss vs Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(epochs)

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, marker='o', color='green', linewidth=2, markersize=6)
plt.title('Test Accuracy vs Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.ylim(0.85, 0.96)  # 聚焦主要变化区间
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(epochs)

# 自动调整布局
plt.tight_layout()
plt.savefig('training_curves_v2.png', dpi=300, bbox_inches='tight')  # 可选：保存图像
plt.show()