import torch
import numpy as np

import torch

def dftmtx_torch(N, device='cpu'):
    """
    生成 N x N 的离散傅里叶变换（DFT）矩阵（基于 PyTorch 实现）

    参数:
        N (int): 矩阵的大小
        device (str): 指定计算设备（如 'cpu' 或 'cuda'）

    返回:
        torch.Tensor: N x N 的 DFT 矩阵
    """
    # 创建一个 NxN 的索引矩阵
    n = torch.arange(N, device=device).float()
    k = n.view(-1, 1)  # 将 n 转换为列向量
    fac = torch.sqrt(torch.tensor(1/N))
    
    # 计算旋转因子
    omega = fac * torch.exp(-2j * torch.pi * k * n / N)
    
    return omega

def dftmtx(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

if __name__ == 'main':
    # 测试生成 4x4 的 DFT 矩阵
    N = 4
    dft_matrix = dftmtx_torch(N)
    print("DFT 矩阵 (PyTorch):")
    print(dft_matrix)
