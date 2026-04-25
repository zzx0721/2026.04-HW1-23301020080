# 2026.04-HW1-23301020080
2026.04计算机视觉HW1: 从零构建三层 MLP 实现 Fashion-MNIST 分类

本项目为计算机视觉课程作业一。本项目从零构建了包含前向传播与反向传播的3层多层感知机（MLP），并在Fashion-MNIST数据集上完成了图像分类任务。

# 环境依赖：
 Python: 3.x 及以上
 
 NumPy: 用于核心的矩阵运算、前向传播与反向传播梯度推导。
 
 Matplotlib: 用于绘制 Loss/Accuracy 学习曲线、混淆矩阵以及权重特征可视化。
 
 标准库: `os`, `gzip`, `urllib.request`, `pickle`, `itertools` 

# 运行训练和测试脚本
  超参数搜索：python search.py
  
  训练：python train.py
  
  测试与评估：python eval.py

# 代码结构
layer.py: 定义了网络层（全连接层 Linear）以及支持切换的激活函数（ReLU, Sigmoid）。

loss.py: 实现了带有数值稳定性优化的 Softmax 交叉熵损失函数（CrossEntropyLoss）。

model.py: 网络组装模块，支持自定义隐藏层维度构建 MLP 模型。

optim.py: 实现了带动量的随机梯度下降（SGD）和阶梯式学习率衰减（StepLR）。

utils.py: 数据处理工具，包含 Fashion-MNIST 的自动下载、解压、归一化、展平以及验证集划分。

train.py: 核心训练脚本，包含完整的训练/验证循环与模型自动保存逻辑。

eval.py: 测试评估脚本，负责加载最优权重、计算测试集准确率并生成可视化图表。

search.py: 超参数搜索脚本，用于寻找最佳的学习率、隐藏层维度和 L2 正则化系数。
