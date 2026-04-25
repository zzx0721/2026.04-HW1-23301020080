# eval.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import MLP
from layer import Linear
from utils import load_fashion_mnist, calculate_accuracy

# 服装类别的文本标签
FASHION_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_model_weights(model, filepath):
    """将保存的权重字典加载回模型实例中"""
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            layer.W = weights[f'layer_{i}_W']
            layer.b = weights[f'layer_{i}_b']
    print(f"成功加载模型权重: {filepath}")


def plot_learning_curves(history_path):
    """绘制训练曲线"""
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', markersize=4)
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='s', markersize=4)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)


def plot_confusion_matrix(y_true, y_pred):
    """手动计算并使用 seaborn 绘制混淆矩阵热图"""
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FASHION_CLASSES, yticklabels=FASHION_CLASSES)
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)


def visualize_weights(model):
    """提取第一层权重并将其可视化为 28x28 的图像"""
    # 找到第一层 Linear 层
    first_layer = [layer for layer in model.layers if isinstance(layer, Linear)][0]
    W = first_layer.W  # shape: (784, hidden_dim)

    # 我们随机挑选 16 个隐藏层神经元的权重进行可视化
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        # 提取第 i 个神经元的权重，并恢复为 28x28 的二维图像空间
        weight_img = W[:, i].reshape(28, 28)
        # 使用 vmin 和 vmax 保证正负权重在颜色上对比明显（如暖色表示正权重，冷色表示负权重）
        im = ax.imshow(weight_img, cmap='RdBu', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
        ax.axis('off')
        ax.set_title(f'Neuron {i}')

    plt.suptitle('Visualization of First Hidden Layer Weights', fontsize=16)
    plt.tight_layout()
    plt.savefig('weight_visualization.png', dpi=300)


def error_analysis(X_test, y_true, y_pred):
    """挑出几个分类错误的样本进行展示"""
    errors = np.where(y_true != y_pred)[0]
    selected_errors = np.random.choice(errors, 5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, idx in enumerate(selected_errors):
        img = X_test[idx].reshape(28, 28)
        true_label = FASHION_CLASSES[y_true[idx]]
        pred_label = FASHION_CLASSES[y_pred[idx]]

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"T: {true_label}\nP: {pred_label}", color='red')

    plt.suptitle('Error Analysis: Misclassified Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300)


def main():
    # 1. 加载测试集数据 (严格遵守相同的归一化处理)
    print("加载测试集...")
    X_test, y_test = load_fashion_mnist(kind='t10k')
    X_test = X_test.astype(np.float32) / 255.0

    # 2. 初始化与训练时完全相同结构的模型
    INPUT_DIM = 784
    HIDDEN_DIMS = [256, 128]
    NUM_CLASSES = 10
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES)

    # 3. 导入最佳权重
    load_model_weights(model, 'checkpoints/best_model.pkl')

    # 4. 测试集前向传播与准确率评估
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)
    test_acc = calculate_accuracy(logits, y_test)
    print(f"[*] 独立测试集分类准确率 (Test Accuracy): {test_acc * 100:.2f}%")

    # 5. 生成报告所需的可视化图表
    print("正在生成图表并保存到本地...")
    plot_learning_curves('checkpoints/history.pkl')
    plot_confusion_matrix(y_test, y_pred)
    visualize_weights(model)
    error_analysis(X_test, y_test, y_pred)
    print("评估完成！所有图表已保存。")


if __name__ == '__main__':
    main()