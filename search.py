# search.py
import itertools
import numpy as np
from model import MLP
from loss import CrossEntropyLoss
from optim import SGD
from utils import load_fashion_mnist, train_val_split, get_batches


def evaluate_model(model, criterion, X_val, y_val):
    """在验证集上评估模型，返回 Accuracy"""
    logits = model.forward(X_val)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_val)
    return acc


def main():
    print("加载数据用于超参数搜索...")
    X_train_full, y_train_full = load_fashion_mnist(kind='train')
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full, val_ratio=0.1)

    # ================================
    # 定义搜索空间 (Grid Search)
    # ================================
    # 每种超参数选2个典型值进行组合 (2x2x2 = 8次实验)
    lr_options = [0.1, 0.01]
    hidden_dims_options = [[256, 128], [128, 64]]
    weight_decay_options = [1e-4, 0.0]  # 加正则化 vs 不加正则化

    epochs_per_search = 5  # 仅跑5个epoch来快速评估潜力
    batch_size = 64

    best_acc = 0.0
    best_params = None
    results_log = []

    print("\n开始网格搜索 (Grid Search)...")
    print("-" * 60)
    print(f"{'LR':<8} | {'Hidden Dims':<15} | {'L2 Decay':<10} | {'Val Acc'}")
    print("-" * 60)

    # 遍历所有参数组合
    for lr, hidden_dims, wd in itertools.product(lr_options, hidden_dims_options, weight_decay_options):

        # 初始化模型
        model = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10, weight_decay=wd)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.get_parameters(), learning_rate=lr)

        # 快速训练几个 Epoch
        for epoch in range(epochs_per_search):
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size, shuffle=True):
                logits = model.forward(X_batch)
                criterion.forward(logits, y_batch)
                d_logits = criterion.backward()
                model.backward(d_logits)
                optimizer.step()

        # 评估当前组合在验证集上的表现
        val_acc = evaluate_model(model, criterion, X_val, y_val)

        # 记录并打印结果
        results_log.append((lr, hidden_dims, wd, val_acc))
        print(f"{lr:<8} | {str(hidden_dims):<15} | {wd:<10} | {val_acc:.4f}")

        # 更新最佳参数
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = (lr, hidden_dims, wd)

    print("-" * 60)
    print(f"搜索完成！最佳参数组合为:")
    print(f"学习率 (LR): {best_params[0]}")
    print(f"隐藏层结构 (Hidden Dims): {best_params[1]}")
    print(f"正则化 (Weight Decay): {best_params[2]}")
    print(f"最高验证集准确率: {best_acc:.4f}")


if __name__ == '__main__':
    main()