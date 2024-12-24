import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


# 频谱图数据集类
class PianoDataset(Dataset):
    def __init__(self, mel_dir, transform=None):
        """初始化数据集，加载梅尔频谱图和标签"""
        self.mel_dir = mel_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小到更小的尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.samples = []
        self.labels = []

        # 遍历频谱图文件夹
        for label_idx, piano_type in enumerate(sorted(os.listdir(mel_dir))):
            piano_path = os.path.join(mel_dir, piano_type)
            if os.path.isdir(piano_path):
                for mel_file in os.listdir(piano_path):
                    if mel_file.endswith('.jpg'):
                        self.samples.append(os.path.join(piano_path, mel_file))
                        self.labels.append(label_idx)

        print(f"Loaded {len(self.samples)} samples from {mel_dir}")

    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取指定索引的样本和标签"""
        mel_path = self.samples[idx]
        try:
            # 读取梅尔频谱图
            mel_img = Image.open(mel_path).convert('L')  # 转换为灰度图

            # 应用变换
            mel_tensor = self.transform(mel_img)

            return mel_tensor, self.labels[idx]

        except Exception as e:
            print(f"Error processing file {mel_path}: {str(e)}")
            # 返回一个空白的频谱图作为替代
            return torch.zeros((1, 224, 224), dtype=torch.float32), self.labels[idx]


# 定义模型
class PianoClassifier(nn.Module):
    def __init__(self, num_classes):
        """初始化模型架构"""
        super(PianoClassifier, self).__init__()

        # 卷积层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第二个卷积块
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.features(x)
        x = self.adaptive_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)
        return x


class ImprovedPianoClassifier(nn.Module):
    def __init__(self, num_classes):
        """初始化模型架构"""
        super(ImprovedPianoClassifier, self).__init__()

        # 卷积层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # 第三个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.features(x)
        x = self.adaptive_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)
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
