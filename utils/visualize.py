import torch
import matplotlib.pyplot as plt

def plot_magnitude(matrix):

    # 计算矩阵的幅值（绝对值）
    magnitude = torch.abs(matrix).cpu().numpy()
    
    # 绘制灰度图
    plt.imshow(magnitude, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title('Magnitude of DFT Matrix')
    plt.show()