# train.py
import os
import pickle
import numpy as np

# 导入之前编写的模块
from layer import Linear
from model import MLP
from loss import CrossEntropyLoss
from optim import SGD, StepLR
from utils import (
    download_fashion_mnist,
    load_fashion_mnist,
    train_val_split,
    get_batches,
    calculate_accuracy
)

# ================================
# 1. 超参数设置 (Hyperparameters)
# ================================
INPUT_DIM = 784  # 28x28 展平
HIDDEN_DIMS = [256, 128]  # 自定义隐藏层结构，可随意修改
NUM_CLASSES = 10  # 服装分类有 10 个类别
BATCH_SIZE = 64  # 批次大小
EPOCHS = 30  # 训练轮数
LEARNING_RATE = 0.1  # 初始学习率
WEIGHT_DECAY = 0.0001  # L2 正则化系数


def save_model(model, filepath):
    """提取网络中所有 Linear 层的 W 和 b"""
    weights = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            weights[f'layer_{i}_W'] = layer.W.copy()
            weights[f'layer_{i}_b'] = layer.b.copy()

    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"[*] 突破最佳验证准确率！模型权重已保存至: {filepath}")


def main():
    # ================================
    # 2. 数据加载与预处理
    # ================================
    download_fashion_mnist()
    X_train_full, y_train_full = load_fashion_mnist(kind='train')

    #数据归一化：将 0~255 的整型像素转换为 0.0~1.0 的浮点数
    X_train_full = X_train_full.astype(np.float32) / 255.0

    # 划分验证集 (保留 10% 数据作为验证集)
    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full, val_ratio=0.1)

    print(f"训练集规模: {X_train.shape[0]} 样本")
    print(f"验证集规模: {X_val.shape[0]} 样本")

    # ================================
    # 3. 初始化模型组件
    # ================================
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, weight_decay=WEIGHT_DECAY)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_parameters(), learning_rate=LEARNING_RATE)

    # 学习率衰减策略：每 10 个 Epoch 学习率乘以 0.5
    scheduler = StepLR(optimizer, decay_step=10, gamma=0.5)

    # ================================
    # 4. 训练与验证循环
    # ================================
    best_val_acc = 0.0
    # 用于记录曲线数据，方便后续写实验报告
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = 'checkpoints/best_model.pkl'

    print("\n开始训练...")
    for epoch in range(1, EPOCHS + 1):

        # --- 训练阶段 ---
        train_losses = []
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE, shuffle=True):
            # 前向传播 (Forward)
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)

            # 反向传播 (Backward)
            d_logits = criterion.backward()
            model.backward(d_logits)

            # 优化器步进更新参数 (Update)
            optimizer.step()

            # 记录当前 batch 的指标
            train_losses.append(loss)
            preds = np.argmax(logits, axis=1)
            train_correct += np.sum(preds == y_batch)
            train_total += len(y_batch)

        # 计算整个 Epoch 的平均训练损失和准确率
        avg_train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total

        # --- 验证阶段 ---
        # 计算前向传播和评估
        val_logits = model.forward(X_val)
        val_loss = criterion.forward(val_logits, y_val)
        val_acc = calculate_accuracy(val_logits, y_val)

        # 将当前 Epoch 数据记录到 history 字典中
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # --- 保存最优模型 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, best_model_path)

        # --- 学习率衰减 ---
        scheduler.step()

    # 训练结束后，保存 history 以便后续在报告中画图
    with open('checkpoints/history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("\n训练全部完成！各项数据及最优模型已保存。")


if __name__ == '__main__':
    main()