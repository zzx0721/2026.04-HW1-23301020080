# optim.py
import numpy as np

class SGD:
    """随机梯度下降优化器"""
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.lr = learning_rate

    def step(self):
        for layer in self.params:
            # 更新权重
            layer.W -= self.lr * layer.dW
            # 更新偏置
            layer.b -= self.lr * layer.db

    def set_lr(self, new_lr):
        self.lr = new_lr


class StepLR:
    """学习率衰减策略 (阶梯式衰减)"""
    def __init__(self, optimizer, decay_step, gamma=0.1):
        self.optimizer = optimizer
        self.decay_step = decay_step
        self.gamma = gamma
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch % self.decay_step == 0:
            new_lr = self.optimizer.lr * self.gamma
            self.optimizer.set_lr(new_lr)
            print(f"--- 学习率衰减为: {new_lr:.6f} ---")