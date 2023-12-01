import pandas as pd
import matplotlib.pyplot as plt
import re

# 读取日志文件
with open('result/cifar10/logs.txt', 'r') as file:
    log_data = file.readlines()

# 解析数据
loss_data = []
acc_data = []
for line in log_data:
    if "Training loss" in line:
        if match := re.search(
            r'\[([0-9]+)/50\], Step: \[([0-9]+)/416\], Training loss: ([0-9.]+), device: cuda:([0-9])',
            line,
        ):
            epoch, step, loss, cuda = match.groups()
            loss_data.append([int(epoch), int(step), float(loss), int(cuda)])
    elif "Val acc" in line:  # 解析准确率数据
        if match := re.search(
            r'Epoch: ([0-9]+), Val acc: ([0-9.]+)',
            line,
        ):
            epoch, acc = match.groups()
            acc_data.append([int(epoch), float(acc)])

# 转换为 DataFrame
df_loss = pd.DataFrame(
    loss_data, columns=['Epoch', 'Step', 'Training Loss', 'Cuda Device'])
df_acc = pd.DataFrame(
    acc_data, columns=['Epoch', 'Accuracy'])

# 计算每个 epoch 的平均准确率
avg_acc_per_epoch = df_acc.groupby('Epoch')['Accuracy'].mean().reset_index()

# 绘制训练损失图
fig, ax1 = plt.subplots()
for device in df_loss['Cuda Device'].unique():
    subset = df_loss[df_loss['Cuda Device'] == device]
    # ax1.plot(subset['Epoch'], subset['Training Loss'], label=f'Cuda:{device} Loss', linewidth=0.5, alpha=0.5)
    ax1.plot(subset['Epoch']+subset['Step']/416, subset['Training Loss'], label=f'Cuda:{device} Loss', linewidth=0.5, alpha=0.5)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.tick_params(axis='y')
ax1.grid(True)  # 添加网格

# 绘制准确率图
ax2 = ax1.twinx()
ax2.plot(avg_acc_per_epoch['Epoch'], avg_acc_per_epoch['Accuracy'], label='Average Validation Accuracy', color='red', linewidth=2)
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y', labelcolor='red')

# 图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title('Training Loss and Average Validation Accuracy over Epochs')
plt.savefig('result/cifar10/loss_and_accuracy.png')
plt.show()

# 输出准确率数据的数量
print(f"Number of average accuracy entries: {len(avg_acc_per_epoch)}")
