# utils.py
import os
import gzip
import urllib.request
import numpy as np


def download_fashion_mnist(data_dir='data'):
    """从官方源自动下载 Fashion-MNIST 数据集"""
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    os.makedirs(data_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"正在下载 {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
    print("Fashion-MNIST 数据集准备就绪！")


def load_fashion_mnist(data_dir='data', kind='train'):
    """
    解析压缩的二进制文件并返回 NumPy 数组
    """
    if kind == 'train':
        labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    else:
        labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')

    # 解析标签 (从第 8 个字节开始)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    # 解析图像 (从第 16 个字节开始，并展平为 784 维)
    with gzip.open(images_path, 'rb') as imgpath:
        # 每张图像是 28x28，因此展平后为 784 维特征向量
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def train_val_split(X, y, val_ratio=0.1, seed=42):
    """划分训练集和验证集"""
    np.random.seed(seed)
    # 打乱索引
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_ratio)

    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def get_batches(X, y, batch_size, shuffle=True):
    """
    数据批次生成器 (Generator)
    使用 yield 可以节省内存，每次只产出一个 batch 的数据
    """
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]


def calculate_accuracy(logits, labels):
    """计算分类准确率"""
    # 找到每一行中最大概率对应的类别索引
    predictions = np.argmax(logits, axis=1)
    # 统计预测正确的比例
    accuracy = np.mean(predictions == labels)
    return accuracy