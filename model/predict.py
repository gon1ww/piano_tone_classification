import torch
import os
from pathlib import Path
import numpy as np
from model import PianoClassifier, PianoDataset
import matplotlib.pyplot as plt
import random

ROOT_DIR = Path(__file__).parent.parent

def load_model(model_path, num_classes):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PianoClassifier(num_classes).to(device)
    
    try:
        # 尝试加载完整的checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    model.eval()
    return model, device

def predict_samples(model, test_dataset, start_idx=0, num_samples=5):
    """预测并展示选定样本的结果"""
    device = next(model.parameters()).device
    
    # 验证参数
    total_samples = len(test_dataset)
    if start_idx < 0 or start_idx >= total_samples:
        print(f"Invalid start_idx. Must be between 0 and {total_samples-1}")
        start_idx = 0
    
    # 确保不会超出数据集范围
    if start_idx + num_samples > total_samples:
        num_samples = total_samples - start_idx
        print(f"Adjusted num_samples to {num_samples} to fit dataset size")
    
    # 生成连续的索引
    indices = list(range(start_idx, start_idx + num_samples))
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i, idx in enumerate(indices):
        # 获取样本
        mel_spectrogram, true_label = test_dataset[idx]
        mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(mel_spectrogram)
            predicted = outputs.argmax(1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted].item()
        
        # 获取原始音频文件路径
        audio_path = test_dataset.samples[idx]
        
        # 绘制梅尔频谱图
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(mel_spectrogram.squeeze().cpu().numpy(), aspect='auto', origin='lower')
        plt.title(f'File: {os.path.basename(audio_path)}\n'
                 f'True Label: {true_label}, Predicted: {predicted}, Confidence: {confidence:.2%}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "model", "picture", "predictions.png"))
    plt.close()
    
    return indices

def evaluate_model(model_path, start_idx=0, num_samples=5):
    """评估模型性能并展示预测结果"""
    # 加载测试数据集
    test_dataset = PianoDataset(os.path.join(ROOT_DIR, "data", "audio", "test"))
    
    # 加载模型
    num_classes = len(set(test_dataset.labels))
    model, device = load_model(model_path, num_classes)
    
    # 预测样本
    indices = predict_samples(model, test_dataset, start_idx, num_samples)
    
    return model, test_dataset, indices

if __name__ == "__main__":
    # 获取用户输入
    print(f"Total test samples: 60 (0-59)")
    start_idx = int(input("Enter start index (0-59): "))
    num_samples = int(input("Enter number of samples to predict: "))
    
    model_path = os.path.join(ROOT_DIR, "model", "best_model.pth")
    model, test_dataset, indices = evaluate_model(model_path, start_idx, num_samples)
    
    # 打印选定样本的详细信息
    print("\nDetailed prediction results:")
    for idx in indices:
        audio_path = test_dataset.samples[idx]
        mel_spectrogram, true_label = test_dataset[idx]
        
        # 预测
        with torch.no_grad():
            outputs = model(mel_spectrogram.unsqueeze(0).to(next(model.parameters()).device))
            predicted = outputs.argmax(1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted].item()
        
        print(f"\nFile: {os.path.basename(audio_path)}")
        print(f"Sample Index: {idx}")
        print(f"True Label: {true_label}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2%}") 