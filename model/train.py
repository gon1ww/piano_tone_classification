import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from modelscope.msdatasets import MsDataset

from model import ImprovedPianoClassifier
from model import PianoClassifier

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# 直接从当前目录导入
from model import PianoDataset, train_model, validate_model
import torch.nn as nn
import torch.optim as optim


def prepare_dataset():
    """下载并准备eval数据集"""
    # 检查数据集是否已存在
    data_dir = os.path.join(ROOT_DIR, "data", "eval")
    if os.path.exists(data_dir):
        print("Dataset already exists, skipping download...")
        return

    # 下载完整的eval数据集
    print("\nLoading eval dataset...")
    ds = MsDataset.load(
        'ccmusic-database/pianos',
        subset_name='eval'
    )

    # 获取数据集总大小
    dataset = list(ds['train'])  # 转换为列表以便访问
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")

    # 计算划分大小
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print(f"\nSplitting dataset into:")
    print(f"Train set: {train_size} samples")
    print(f"Validation set: {val_size} samples")
    print(f"Test set: {test_size} samples")

    # 创建必要的目录
    splits = {
        'train': (0, train_size),
        'validation': (train_size, train_size + val_size),
        'test': (train_size + val_size, total_size)
    }

    for split_name, (start_idx, end_idx) in splits.items():
        split_dir = os.path.join(ROOT_DIR, "data", "eval", split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\nProcessing {split_name} split...")

        # 获取该划分的数据
        split_data = dataset[start_idx:end_idx]
        total_items = len(split_data)
        print(f"Processing {total_items} items in {split_name} split")

        for idx, item in enumerate(split_data):
            try:
                mel_img = item['mel']  # 获取梅尔频谱图PIL图像对象
                label = item['label']
                label_dir = os.path.join(split_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)

                # 保存梅尔频谱图
                filename = f"mel_{idx}.jpg"
                save_path = os.path.join(label_dir, filename)
                mel_img.save(save_path, "JPEG")

                # 打印进度
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{total_items} items in {split_name} split")

            except Exception as e:
                print(f"Error processing item {idx} in {split_name} split: {str(e)}")
                print(f"Item content: {item}")
                continue

    print("\nDataset preparation completed!")


def load_dataset_splits():
    # 读取数据集划分信息
    with open(os.path.join(ROOT_DIR, "dataset_infos.json"), "r") as f:
        dataset_info = json.load(f)
    return dataset_info["eval"]["splits"]  # 使用eval数据集的划分信息


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


def select_model_type():
    """让用户选择要使用的模型类型"""
    while True:
        print("\nSelect Model Type:")
        print("1. PianoClassifier (Basic Model)")
        print("2. ImprovedPianoClassifier (Advanced Model)")
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '1':
            return PianoClassifier, "piano_classifier"
        elif choice == '2':
            return ImprovedPianoClassifier, "improved_piano_classifier"
        else:
            print("Invalid choice. Please try again.")


def get_num_epochs():
    """让用户输入训练轮数"""
    while True:
        try:
            epochs = int(input("\nEnter number of epochs for training (1-50): "))
            if 1 <= epochs <= 50:
                return epochs
            else:
                print("Please enter a number between 1 and 50.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 准备数据集
    print("Preparing dataset...")
    prepare_dataset()

    stop = input("Press Enter to continue...")
    # 选择模型类型
    model_class, model_name = select_model_type()
    print(f"\nSelected model: {model_name}")

    # 获取训练轮数
    num_epochs = get_num_epochs()
    print(f"\nTraining for {num_epochs} epochs")

    # 获取数据集划分信息
    splits = load_dataset_splits()

    # 加载训练集、验证集和测试集
    train_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "eval", "train"))
    val_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "eval", "validation"))
    test_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "eval", "test"))

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # 初始化模型
    num_classes = len(set(train_dataset.labels))
    model = model_class(num_classes).to(device)
    print(f"\nInitialized {model_class.__name__} with {num_classes} classes")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    best_val_acc = 0.0

    # 添加列表来记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

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
            # 保存模型时包含模型类型信息
            model_save_path = os.path.join(ROOT_DIR, "model", f"best_{model_name}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    # 在测试集上评估最终模型
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
