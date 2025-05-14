import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换颜色空间
img_bgr = cv2.imread(r'dataset\ROCK\PPL-\a13_1.png')  # 替换为您的图像路径
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 提取LAB通道
L, a, b = cv2.split(img_lab)

# 提取HSV通道
H, S, V = cv2.split(img_hsv)

# 可视化设置
plt.figure(figsize=(18, 10))

# 显示原始RGB图像
plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title('Original RGB')
plt.axis('off')

# 显示LAB通道
plt.subplot(2, 4, 2)
plt.imshow(L, cmap='gray')
plt.title('L Channel (Lightness)')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(a, cmap='RdYlGn')  # 使用红-黄-绿色彩映射增强对比
plt.title('a Channel (Green-Red)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(b, cmap='RdYlBu')  # 使用红-黄-蓝色彩映射增强对比
plt.title('b Channel (Blue-Yellow)')
plt.axis('off')

# 显示HSV通道
plt.subplot(2, 4, 6)
plt.imshow(H, cmap='hsv')  # 使用HSV循环色相映射
plt.title('H Channel (Hue)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(S, cmap='gray')
plt.title('S Channel (Saturation)')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(V, cmap='gray')
plt.title('V Channel (Value)')
plt.axis('off')

plt.tight_layout()
plt.show()
