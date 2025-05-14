import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
# rcParams['font.family'] = 'Times New Roman'

def load_loss_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    epochs = [x[1] + 1 for x in data]
    losses = [x[2] for x in data]
    return np.array(epochs), np.array(losses)

# 加载数据
train_epochs, train_losses = load_loss_data('tbd_runs_data_loss_trn_loss.json')
test_epochs, test_losses = load_loss_data('tbd_runs_data_loss_val_loss.json')

# 创建图表（明确设置背景色）
plt.figure(figsize=(10, 6), facecolor='white', dpi=100)
ax = plt.gca()
ax.set_facecolor('white')

# 绘制曲线（带填充效果的标记）
train_line, = plt.plot(train_epochs, train_losses, 
                      label='Training Loss', 
                      color='#1f77b4',
                      marker='o',
                      linestyle='-',
                      linewidth=2.5,
                      markersize=8,
                      markevery=5,
                      markerfacecolor='white',
                      markeredgecolor='#1f77b4',
                      markeredgewidth=1.5)

test_line, = plt.plot(test_epochs, test_losses,
                     label='Validation Loss',
                     color='#d62728',
                     marker='s',
                     linestyle='-',
                     linewidth=2.5,
                     markersize=8,
                     markevery=5,
                     markerfacecolor='white',
                     markeredgecolor='#d62728',
                     markeredgewidth=1.5)

# 标题和标签（增加字体层次）
plt.title('Model Training Progress', 
         fontsize=15, pad=20, 
         fontweight='bold', color='black')
plt.xlabel('Epoch', fontsize=13, labelpad=12, fontweight='semibold' )
plt.ylabel('Loss Value', fontsize=13, labelpad=12, fontweight='semibold')

# 坐标轴封闭处理（显示四个边框）
ax.spines['bottom'].set_position(('axes', 0.0))  # 固定到底部
ax.spines['left'].set_position(('axes', 0.0))    # 固定到左侧
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.2)

# 刻度设置（更精细控制）
ax.tick_params(axis='both', which='major', 
              length=6, width=1.2,
              labelsize=12, pad=8,
              color='black',
              direction='in')
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.set_axisbelow(True)  # 网格线在数据下方（如果启用）

# 图例美化
legend = plt.legend(fontsize=12,
                   frameon=True,
                   loc='upper right',
                   facecolor='white',
                   edgecolor='gray',
                   framealpha=1,
                   borderaxespad=0.8,
                   handletextpad=0.5)
legend.get_frame().set_linewidth(1.2)

# 网格线（按需启用）
plt.grid(False)

# 调整边距并保存
plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
plt.savefig('loss_curve_pro.png', 
           dpi=300, 
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')
plt.show()
