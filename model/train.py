import torch
from torch.utils.data import random_split
import sys
import os
from pathlib import Path
import json
import shutil
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# 直接从当前目录导入
from model import PianoDataset, PianoClassifier, train_model, validate_model
import torch.nn as nn
import torch.optim as optim


def prepare_dataset():
    """下载并准备数据集"""
    # 检查数据集是否已存在
    data_dir = os.path.join(ROOT_DIR, "data", "audio")
    if os.path.exists(data_dir):
        print("Dataset already exists, skipping download...")
        return

    # 下载数据集
    ds = MsDataset.load(
        'ccmusic-database/pianos',
        subset_name='eval'
    )

    # 创建必要的目录
    splits = ['train', 'validation', 'test']

    for split in splits:
        split_dir = os.path.join(ROOT_DIR, "data", "eval", split)
        os.makedirs(split_dir)

        # 将音频文件移动到对应目录
        split_data = ds[split]  # 获取对应的数据集划分

        for item in split_data:
            try:
                mel_path = item['mel']['path']
                label = item['label']
                label_dir = os.path.join(split_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)

                filename = os.path.basename(mel_path)
                shutil.copy2(mel_path, os.path.join(label_dir, filename))

            except Exception as e:
                print(f"Error processing item: {item}")
                print(f"Error message: {str(e)}")
                continue


def load_dataset_splits():
    # 读取数据集划分信息
    with open(os.path.join(ROOT_DIR, "dataset_infos.json"), "r") as f:
        dataset_info = json.load(f)
    return dataset_info["default"]["splits"]


def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练过程的loss和accuracy曲线"""
    plt.figure(figsize=(12, 5))

    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "model", "picture", "training_curves.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, filename='confusion_matrix.png'):
    """绘制混淆矩阵并保存为PNG文件"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(ROOT_DIR, "model", "picture", filename))
    plt.close()


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 准备数据集
    print("Preparing dataset...")
    prepare_dataset()
    a = input("Press Enter to continue...")

    # 获取数据集划分信息
    splits = load_dataset_splits()

    # 加载训练集、验证集和测试集
    train_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "mel_spectrogram", "train"))
    val_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "mel_spectrogram", "validation"))
    test_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "mel_spectrogram", "test"))

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,  # 批量大小
        shuffle=True,
        num_workers=2,  # 工作进程数
        pin_memory=True,
        persistent_workers=True,  # 保持工作进程存活
        prefetch_factor=2  # 预取数量
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型
    num_classes = len(set(train_dataset.labels))
    model = PianoClassifier(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20
    best_val_acc = 0.0

    # 添加列表来记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 用于记录所有的真实标签和预测标签
    all_true_labels = []
    all_pred_labels = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        # 记录训练过程
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ROOT_DIR, "model", "best_model.pth"))
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")

        # 在验证集上进行预测以生成混淆矩阵
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted.cpu().numpy())

    # 绘制混淆矩阵
    plot_confusion_matrix(all_true_labels, all_pred_labels, classes=list(range(num_classes)))

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    # 在测试集上评估最终模型
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main() 