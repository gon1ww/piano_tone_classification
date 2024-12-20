import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf

# 音频数据集类
class PianoDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        """初始化数据集，加载音频文件和标签"""
        self.audio_dir = audio_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # 遍历音频文件夹
        for label_idx, piano_type in enumerate(sorted(os.listdir(audio_dir))):
            piano_path = os.path.join(audio_dir, piano_type)
            if os.path.isdir(piano_path):
                for audio_file in os.listdir(piano_path):
                    if audio_file.endswith('.wav'):
                        self.samples.append(os.path.join(piano_path, audio_file))
                        self.labels.append(label_idx)
        
        print(f"Loaded {len(self.samples)} samples from {audio_dir}")
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本和标签"""
        audio_path = self.samples[idx]
        try:
            # 使用 sf.read 替代 torchaudio.load，可以更好地控制内存
            waveform, sample_rate = sf.read(audio_path, dtype='float32')
            
            # 转换为 torch tensor 并调整维度
            waveform = torch.from_numpy(waveform.T)
            
            # 如果是双声道，转换为单声道
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 限制最大长度，避免过长的音频
            max_length = 220500  # 5秒 * 44100 采样率
            if waveform.shape[1] > max_length:
                start = torch.randint(0, waveform.shape[1] - max_length, (1,))
                waveform = waveform[:, start:start + max_length]
            else:
                # 填充到固定长度
                pad_length = max_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_length))
            
            # 提取梅尔频谱图特征
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,  # 减小 FFT 大小
                hop_length=256,  # 减小跳跃长度
                n_mels=64,  # 减小梅尔频带数
                power=1.0  # 使用幅度谱而不是功率谱
            )(waveform)
            
            # 转换为分贝单位并规范化
            mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (mel_spectrogram.std() + 1e-8)
            
            if self.transform:
                mel_spectrogram = self.transform(mel_spectrogram)
            
            # 确保输入维度正确
            mel_spectrogram = mel_spectrogram.squeeze()
            if len(mel_spectrogram.shape) == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)
            
            # 转换为 float32 以节省内存
            mel_spectrogram = mel_spectrogram.float()
            
            return mel_spectrogram, self.labels[idx]
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")
            # 返回一个空白的频谱图作为替代
            return torch.zeros((1, 64, 172), dtype=torch.float32), self.labels[idx]

# 定义模型

# class PianoClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(PianoClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, num_classes)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.adaptive_pool(x)
#         x = x.view(-1, 64 * 8 * 8)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

class PianoClassifier(nn.Module):
    def __init__(self, num_classes):
        """初始化模型架构"""
        super(PianoClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # 减小全连接层的维度
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),  # 减小隐藏层维度
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)  # 减小dropout率
        )
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """前向传播"""
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    """训练模型并返回损失和准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

# 验证函数
def validate_model(model, val_loader, criterion, device):
    """验证模型并返回损失和准确率"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total 