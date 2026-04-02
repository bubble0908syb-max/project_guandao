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

# ==========================================
# 1. 超参数与配置
# ==========================================
DATA_DIR = './Pipeline_Denoised_Data'  # 使用VMD降噪后的数据
RESULTS_DIR = './Pipeline_CNN2D_Results'
SEQ_LENGTH = 1024  # 每个样本的序列长度
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# 类别映射字典
CLASS_MAP = {
    'no Leak': 0,
    '0.4mm leak': 1,
    '2mm leak': 2,
    '4mm leak': 3
}

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# ==========================================
# 2. 一维信号转二维图像核心函数 (STFT)
# ==========================================
def signal_to_image(segment, fs=1000):
    """
    升级版：使用对数短时傅里叶变换 (Log-STFT)
    """
    # 1. 增大 nperseg，提高频率轴的分辨率
    frequencies, times, Zxx = signal.stft(segment, fs=fs, nperseg=128, noverlap=64)
    magnitude_spectrogram = np.abs(Zxx)

    # 2. 关键：转换为对数刻度 (Log scale)
    # 加 1e-9 是为了防止对数为负无穷
    log_spectrogram = 10 * np.log10(magnitude_spectrogram + 1e-9)

    # 3. Min-Max 归一化到 [0, 1] 区间
    min_val = np.min(log_spectrogram)
    max_val = np.max(log_spectrogram)
    if max_val > min_val:
        normalized_img = (log_spectrogram - min_val) / (max_val - min_val)
    else:
        normalized_img = log_spectrogram

    return normalized_img


# ==========================================
# 3. 数据加载与预处理
# ==========================================
def load_and_transform_data(data_dir, seq_length=1024):
    X_images = []
    y_labels = []

    print("开始加载数据并进行 1D -> 2D 转换...")
    for class_name, label in CLASS_MAP.items():
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.exists(folder_path):
            print(f"警告: 找不到文件夹 {folder_path}")
            continue

        files = glob.glob(os.path.join(folder_path, '*.csv'))
        class_samples = 0

        for file in files:
            try:
                # 读取降噪后的CSV (假设信号在第一列或第二列)
                df = pd.read_csv(file)
                # 获取最后数值列作为信号
                sig = df.iloc[:, -1].values

                # 滑动窗口切片
                num_segments = len(sig) // seq_length
                for i in range(num_segments):
                    segment = sig[i * seq_length: (i + 1) * seq_length]

                    # 关键步骤：1D转2D
                    img2d = signal_to_image(segment)

                    X_images.append(img2d)
                    y_labels.append(label)
                    class_samples += 1
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        print(f"类别 '{class_name}' (标签 {label}) 提取了 {class_samples} 个时频图样本.")

    # 转换为 numpy 数组，增加通道维度 (N, C, H, W) -> (N, 1, H, W)
    X_images = np.array(X_images)
    X_images = np.expand_dims(X_images, axis=1)
    y_labels = np.array(y_labels)

    print(f"\n数据集转换完成! 总样本数: {len(y_labels)}, 图像维度: {X_images.shape}")
    return X_images, y_labels


class LeakageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 4. 构建 2D CNN 模型
# ==========================================
class PipelineCNN2D(nn.Module):
    def __init__(self, num_classes=4):
        super(PipelineCNN2D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 将 (4, 4) 改为 (8, 8)，保留更多空间时频特征
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 这里对应修改为 64 * 8 * 8
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==========================================
# 5. 可视化STFT样本 (可选，用于论文/报告展示)
# ==========================================
def plot_sample_spectrogram(X, y):
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(y)
    inv_map = {v: k for k, v in CLASS_MAP.items()}

    for i, label in enumerate(unique_labels):
        idx = np.where(y == label)[0][0]  # 找该类别的第一个样本
        img = X[idx][0]  # 移除通道维度

        plt.subplot(2, 2, i + 1)
        plt.imshow(img, aspect='auto', origin='lower', cmap='jet')
        plt.title(f"Spectrogram: {inv_map[label]}")
        plt.colorbar(label='Magnitude')
        plt.ylabel('Frequency Bin')
        plt.xlabel('Time Step')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Sample_Spectrograms.png'), dpi=300)
    plt.close()
    print(f"时频图样本已保存至 {RESULTS_DIR}/Sample_Spectrograms.png")


# ==========================================
# 6. 训练与评估流程
# ==========================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")

    # 1. 准备数据
    X, y = load_and_transform_data(DATA_DIR, SEQ_LENGTH)
    if len(X) == 0:
        print("未加载到数据，请检查路径。")
        return

    plot_sample_spectrogram(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = LeakageDataset(X_train, y_train)
    test_dataset = LeakageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = PipelineCNN2D(num_classes=len(CLASS_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 训练循环
    train_losses, test_accuracies = [], []
    best_acc = 0.0

    print("\n开始训练 2D CNN 模型...")
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

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 验证评估
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
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_cnn2d.pth'))
            best_preds = all_preds
            best_labels = all_labels

    print(f"\n训练结束！最佳测试集准确率: {best_acc:.4f}")

    # 4. 生成分析报告和图表
    target_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda item: item[1])]

    # 分类报告
    report = classification_report(best_labels, best_preds, target_names=target_names)
    print("分类报告:")
    print(report)
    with open(os.path.join(RESULTS_DIR, 'CNN2D_Metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Best Accuracy: {best_acc}\n\n{report}")

    # 绘制混淆矩阵
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('2D CNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'CNN2D_Confusion_Matrix.png'), dpi=300)
    plt.close()

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Validation Accuracy', color='blue')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'CNN2D_Training_Curves.png'), dpi=300)
    plt.close()

    print(f"所有结果与可视化图表已保存至: {RESULTS_DIR}")


if __name__ == '__main__':
    train_and_evaluate()