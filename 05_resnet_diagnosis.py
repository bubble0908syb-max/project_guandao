import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision.models as models

# ==========================================
# 1. 超参数与配置
# ==========================================
DATA_DIR = './Pipeline_Denoised_Data'
RESULTS_DIR = './Pipeline_ResNet_Results'
SEQ_LENGTH = 1024
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

CLASS_MAP = {
    'no Leak': 0,
    '0.4mm leak': 1,
    '2mm leak': 2,
    '4mm leak': 3
}

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# ==========================================
# 2. 信号转二维图像 (保留最优的 Log-STFT 方案)
# ==========================================
def signal_to_image(segment, fs=1000):
    """
    使用对数短时傅里叶变换 (Log-STFT) 提取高频微弱特征
    """
    frequencies, times, Zxx = signal.stft(segment, fs=fs, nperseg=128, noverlap=64)
    magnitude_spectrogram = np.abs(Zxx)

    # 转换为对数刻度，增强微弱的泄漏高频特征
    log_spectrogram = 10 * np.log10(magnitude_spectrogram + 1e-9)

    # 归一化到 [0, 1]
    min_val = np.min(log_spectrogram)
    max_val = np.max(log_spectrogram)
    if max_val > min_val:
        normalized_img = (log_spectrogram - min_val) / (max_val - min_val)
    else:
        normalized_img = log_spectrogram

    return normalized_img


# ==========================================
# 3. 数据加载
# ==========================================
def load_and_transform_data(data_dir, seq_length=1024):
    X_images, y_labels = [], []

    print("开始加载数据并进行 Log-STFT 转换...")
    for class_name, label in CLASS_MAP.items():
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.exists(folder_path):
            continue

        files = glob.glob(os.path.join(folder_path, '*.csv'))
        class_samples = 0

        for file in files:
            try:
                df = pd.read_csv(file)
                sig = df.iloc[:, -1].values
                num_segments = len(sig) // seq_length

                for i in range(num_segments):
                    segment = sig[i * seq_length: (i + 1) * seq_length]
                    img2d = signal_to_image(segment)
                    X_images.append(img2d)
                    y_labels.append(label)
                    class_samples += 1
            except Exception as e:
                pass

        print(f"类别 '{class_name}' 提取了 {class_samples} 个时频图样本.")

    X_images = np.expand_dims(np.array(X_images), axis=1)  # (N, 1, H, W)
    y_labels = np.array(y_labels)
    print(f"数据集准备完毕! 图像维度: {X_images.shape}")
    return X_images, y_labels


class LeakageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ==========================================
# 4. 构建定制版 ResNet 模型
# ==========================================
class PipelineResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(PipelineResNet, self).__init__()

        # 加载基础的 ResNet18 架构 (不使用预训练权重，因为自然图像和时频图特征差异大)
        self.resnet = models.resnet18(pretrained=False)

        # 【核心修改 1】：适应单通道输入
        # 官方 ResNet 默认输入是 RGB 3通道图像。我们的时频图是 1 通道灰度图。
        # 因此必须替换掉第一层卷积 (conv1)。
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # 修改为1通道
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # 【核心修改 2】：适应我们的类别数
        # 官方 ResNet 默认输出 1000 类。这里替换为我们的 4 类。
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.4),  # 添加 Dropout 防止过拟合
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# ==========================================
# 5. 训练与评估流程
# ==========================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[部署环境] 使用设备: {device}")

    X, y = load_and_transform_data(DATA_DIR, SEQ_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_loader = DataLoader(LeakageDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(LeakageDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = PipelineResNet(num_classes=len(CLASS_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    # ResNet 参数较多，添加少量的 weight_decay (L2正则化) 帮助稳定训练
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_losses, test_accuracies = [], []
    best_acc = 0.0

    print("\n🚀 开始训练 ResNet-18 模型...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Val Accuracy: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_resnet.pth'))
            best_preds, best_labels = all_preds, all_labels

    print(f"\n🎉 训练结束！ResNet 最佳测试集准确率: {best_acc:.4f}")

    # 保存报告
    target_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
    report = classification_report(best_labels, best_preds, target_names=target_names)
    with open(os.path.join(RESULTS_DIR, 'ResNet_Metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Best Accuracy: {best_acc}\n\n{report}")

    # 混淆矩阵图
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.title('ResNet Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ResNet_Confusion_Matrix.png'), dpi=300)
    plt.close()

    # 训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.title('ResNet Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Validation Accuracy', color='green')
    plt.title('ResNet Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ResNet_Training_Curves.png'), dpi=300)
    plt.close()
    print(f"所有结果与图表已保存至: {RESULTS_DIR}")


if __name__ == '__main__':
    train_and_evaluate()