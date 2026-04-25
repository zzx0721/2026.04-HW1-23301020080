# loss.py
import numpy as np


class CrossEntropyLoss:
    """交叉熵损失结合 Softmax"""

    def __init__(self):
        self.cache = None

    def forward(self, logits, labels):
        """
        logits: 未经过 softmax 的网络输出，shape: (batch_size, num_classes)
        labels: 真实标签的整数索引，shape: (batch_size,)
        """
        batch_size = logits.shape[0]

        # 1. 每行减去该行的最大值
        # keepdims=True 保证 max_logits shape 为 (batch_size, 1) 以便广播
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)

        # 2. 计算 Softmax 概率
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 缓存 probs 和 labels，供反向传播使用
        self.cache = (probs, labels)

        # 3. 计算交叉熵 Loss
        # 加上 1e-9 是为了防止 log(0) 导致数值异常
        # probs[range(batch_size), labels] 利用高级索引直接提取正确类别的预测概率
        corect_logprobs = -np.log(probs[range(batch_size), labels] + 1e-9)
        loss = np.mean(corect_logprobs)

        return loss

    def backward(self):
        """
        计算损失函数对 logits 的梯度 (包含 Softmax 的求导)
        数学推导结果：d_logits = probs - one_hot(labels)
        """
        probs, labels = self.cache
        batch_size = probs.shape[0]

        # 复制一份 probs，避免直接修改缓存
        d_logits = probs.copy()

        # 根据推导公式，只需在正确类别的概率上减去 1
        d_logits[range(batch_size), labels] -= 1

        # 除以 batch_size，确保梯度的规模不受 batch 大小的影响
        d_logits /= batch_size

        return d_logits