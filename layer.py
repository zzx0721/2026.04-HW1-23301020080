import numpy as np


class Linear:
    """全连接层 """

    def __init__(self, input_dim, output_dim, weight_decay=0.0):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.weight_decay = weight_decay

        self.cache = None
        self.dW = None
        self.db = None

    #前向传播
    def forward(self, X):
        #Z = XW + b
        self.cache = X
        Z=np.dot(X,self.W)+self.b
        return Z

    #反向传播
    def backward(self, dZ):
        """
        dZ: 来自后续层的梯度，shape: (batch_size, output_dim)
        返回: dX, 传递给上一层的梯度
        """
        # 从 forward 缓存中取出输入 X
        X = self.cache

        # 1. 计算 dW (加上 L2 正则化的梯度)
        self.dW = np.dot(X.T, dZ) + self.weight_decay * self.W

        # 2. 计算 db (沿 batch 维度求和)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # 3. 计算传递给上一层的 dX
        dX = np.dot(dZ, self.W.T)

        return dX


class ReLU:
    """ReLU 激活函数"""

    def __init__(self):
        self.cache = None

    def forward(self, Z):
        self.cache = Z
        A=np.maximum(0,Z)
        return A

    def backward(self, dA):
        """dA 为上一层传回的梯度"""
        Z=self.cache
        dZ=dA * (Z>0).astype(float)
        return dZ

class Sigmoid:
    """Sigmoid 激活函数"""
    def __init__(self):
        self.cache = None

    def forward(self, Z):
        self.cache = Z
        # 防止 exp 溢出的稳定性处理
        Z_safe = np.clip(Z, -500, 500)
        A = 1.0 / (1.0 + np.exp(-Z_safe))
        return A

    def backward(self, dA):
        """dA 为上一层传回的梯度"""
        Z = self.cache
        Z_safe = np.clip(Z, -500, 500)
        A = 1.0 / (1.0 + np.exp(-Z_safe))
        # Sigmoid 的导数是 A * (1 - A)
        dZ = dA * A * (1.0 - A)
        return dZ