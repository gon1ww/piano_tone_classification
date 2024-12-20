import torch
import os
from pathlib import Path
import numpy as np
from model import ImprovedPianoClassifier, PianoDataset
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent

def load_model(model_path, num_classes):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedPianoClassifier(num_classes).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    model.eval()
    return model, device

def predict_single_sample(model, test_dataset, idx):
    """预测单个样本"""
    device = next(model.parameters()).device
    
    # 获取样本
    mel_spectrogram, true_label = test_dataset[idx]
    mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(mel_spectrogram)
        predicted = outputs.argmax(1).item()
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted].item()
    
    return {
        'mel_spectrogram': mel_spectrogram,
        'true_label': true_label,
        'predicted_label': predicted,
        'confidence': confidence,
        'file_path': test_dataset.samples[idx]
    }

def visualize_predictions(predictions, save_path=None):
    """可视化预测结果"""
    num_samples = len(predictions)
    plt.figure(figsize=(15, 4*num_samples))
    
    for i, pred in enumerate(predictions):
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(pred['mel_spectrogram'].squeeze().cpu().numpy(), aspect='auto', origin='lower')
        
        # 设置标题，包含预测信息
        title = f'File: {os.path.basename(pred["file_path"])}\n'
        title += f'True Label: {pred["true_label"]}, '
        title += f'Predicted: {pred["predicted_label"]}, '
        title += f'Confidence: {pred["confidence"]:.2%}'
        
        # 根据预测是否正确设置标题颜色
        color = 'green' if pred['true_label'] == pred['predicted_label'] else 'red'
        plt.title(title, color=color)
        plt.colorbar()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    plt.show()
    plt.close()

def evaluate_test_samples():
    """评估测试集样本"""
    # 加载测试数据集
    test_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "eval", "test"))
    total_samples = len(test_dataset)
    
    # 加载模型
    num_classes = len(set(test_dataset.labels))
    model_path = os.path.join(ROOT_DIR, "model", "best_model.pth")
    model, device = load_model(model_path, num_classes)
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 收集所有预测结果用于混淆矩阵
    all_true_labels = []
    all_pred_labels = []
    
    # 在测试集上进行预测
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating test set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 收集标签用于混淆矩阵
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
            
            # 计算准确率
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # 计算总体准确率
    accuracy = 100. * correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix on Test Set')
    
    # 保存混淆矩阵
    save_path = os.path.join(ROOT_DIR, "model", "picture", "test_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return model, test_dataset

if __name__ == "__main__":
    # 评估测试集并生成混淆矩阵
    model, test_dataset = evaluate_test_samples()
    
    # 继续其他交互式评估...
    while True:
        print("\nTest Set Evaluation Options:")
        print("1. View random samples")
        print("2. View specific samples by indices")
        print("3. View samples by label")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            num_samples = int(input("Enter number of random samples to view: "))
            indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
        elif choice == '2':
            indices_input = input("Enter sample indices (comma-separated): ")
            indices = [int(idx) for idx in indices_input.split(',')]
        elif choice == '3':
            label = int(input("Enter label to view: "))
            label_indices = [i for i, l in enumerate(test_dataset.labels) if l == label]
            num_samples = int(input(f"Found {len(label_indices)} samples with label {label}. How many to view? "))
            indices = random.sample(label_indices, min(num_samples, len(label_indices)))
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")
            continue
            
        # 获取预测结果
        predictions = [predict_single_sample(model, test_dataset, idx) for idx in indices]
        
        # 可视化结果
        save_path = os.path.join(ROOT_DIR, "model", "picture", "predictions.png")
        visualize_predictions(predictions, save_path) 