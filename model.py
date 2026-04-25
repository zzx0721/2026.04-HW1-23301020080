# model.py
from layer import Linear, ReLU, Sigmoid

class MLP:
    """多层感知机分类器"""

    def __init__(self, input_dim, hidden_dims, num_classes, weight_decay=0.0):
        """
        input_dim: 输入特征维度 (对于展平的 Fashion-MNIST 是 28*28 = 784)
        hidden_dims: 一个列表，包含每个隐藏层的神经元数量，例如 [128, 64] 表示两个隐藏层
        num_classes: 输出类别数 (Fashion-MNIST 是 10)
        weight_decay: L2 正则化系数
        """
        self.layers = []

        # 1. 构建第一层 (Input -> Hidden 1)
        self.layers.append(Linear(input_dim, hidden_dims[0], weight_decay))
        self.layers.append(ReLU())

        # 2. 构建后续隐藏层 (Hidden i -> Hidden i+1)
        for i in range(1, len(hidden_dims)):
            self.layers.append(Linear(hidden_dims[i - 1], hidden_dims[i], weight_decay))
            self.layers.append(ReLU())

        # 3. 构建输出层 (Hidden last -> Output)
        # 注意：输出层后面直接接 Loss 算 Softmax，所以这里绝对不能加激活函数！
        self.layers.append(Linear(hidden_dims[-1], num_classes, weight_decay))

    def forward(self, X):
        """
        依次经过所有层进行前向传播
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # 这里的 out 就是未经过 Softmax 的 logits

    def backward(self, d_logits):
        """
        依次逆序经过所有层进行反向传播
        d_logits 是由 CrossEntropyLoss 的 backward 传回来的梯度
        """
        dout = d_logits
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_parameters(self):
        """
        提取模型中所有需要更新参数的层 (Linear层)
        为了后续交给 SGD 优化器使用
        """
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append(layer)
        return params