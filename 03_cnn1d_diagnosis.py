import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DENOISED_DIR = './Pipeline_Denoised_Data'
SAVE_DIR = './Pipeline_CNN_Results'
os.makedirs(SAVE_DIR, exist_ok=True)
CLASSES = ['0.4mm leak', '2mm leak', '4mm leak', 'no Leak']
LABEL_MAP = {CLASSES[0]: 0, CLASSES[1]: 1, CLASSES[2]: 2, CLASSES[3]: 3}
CHUNK_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineCNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super(PipelineCNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, 2, 32), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 32, 1, 16), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 16, 1, 8), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2)
        )
        # 💡 核心杀手锏：加入全局平均池化层 (GAP)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 💡 分类器极大简化，参数量骤降，彻底锁死过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # 从原来的 64*64 直接变成了 64
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)  # (Batch, 64, Length) -> (Batch, 64, 1)
        x = x.view(x.size(0), -1)  # 展平后只有 64 维
        return self.classifier(x)

def main():
    print("1. 正在读取去噪后的一维时序数据...")
    frames, labels = [], []
    for c_name in CLASSES:
        files = glob.glob(os.path.join(DENOISED_DIR, c_name, '*.csv'))
        for f in files:
            signal = pd.read_csv(f, header=None).values.flatten()
            for j in range(len(signal) // CHUNK_SIZE):
                frames.append(signal[j * CHUNK_SIZE: (j + 1) * CHUNK_SIZE])
                labels.append(LABEL_MAP[c_name])

    X, y = np.array(frames), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).unsqueeze(1)
    X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_scaled, y_train_t), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_scaled, y_test_t), batch_size=64, shuffle=False)

    model = PipelineCNN1D().to(device)
    criterion = nn.CrossEntropyLoss()
    model = PipelineCNN1D().to(device)
    criterion = nn.CrossEntropyLoss()

    # 💡 使用初始 0.001 的学习率，并保留权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 💡 动态学习率衰减：每 10 个 epoch，学习率减半 (抚平后期的震荡)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"2. 开始训练 1D-CNN (使用 {device})...")
    epochs, best_acc = 100, 0.0
    best_model_path = os.path.join(SAVE_DIR, 'best_cnn1d.pth')
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    start_t = time.time()
    for epoch in range(epochs):
        model.train()
        r_loss, c_train, t_train = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * inputs.size(0)
            c_train += (outputs.argmax(1) == targets).sum().item()
            t_train += targets.size(0)

        train_losses.append(r_loss / t_train);
        train_accs.append(c_train / t_train)

        model.eval()
        r_test_loss, c_test, t_test = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                r_test_loss += criterion(outputs, targets).item() * inputs.size(0)
                c_test += (outputs.argmax(1) == targets).sum().item()
                t_test += targets.size(0)

        ep_acc = c_test / t_test
        test_losses.append(r_test_loss / t_test);
        test_accs.append(ep_acc)

        if ep_acc > best_acc:
            best_acc = ep_acc
            torch.save(model.state_dict(), best_model_path)
            mark = "⭐"
        else:
            mark = ""

        if (epoch + 1) % 5 == 0 or mark:
            print(
                f"Epoch [{epoch + 1}/{epochs}] Train Acc: {train_accs[-1]:.2%} | Test Acc: {test_accs[-1]:.2%} {mark}")
            # 在每个 epoch 结束时，更新一次学习率
            scheduler.step()

    print(f"训练结束! 耗时: {time.time() - start_t:.2f}秒")

    plt.figure(figsize=(14, 5));
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train');
    plt.plot(range(1, epochs + 1), test_losses, label='Test');
    plt.title('Loss');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train');
    plt.plot(range(1, epochs + 1), test_accs, label='Test');
    plt.title('Accuracy');
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'CNN1D_Curves.png'), dpi=150);
    plt.close()

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            all_preds.extend(model(inputs.to(device)).argmax(1).cpu().numpy())

    final_acc = accuracy_score(y_test, all_preds)
    print(f"🏆 1D-CNN 最佳模型测试准确率: {final_acc:.2%}")

    pd.DataFrame(classification_report(y_test, all_preds, target_names=CLASSES, output_dict=True)).transpose().to_csv(
        os.path.join(SAVE_DIR, 'CNN_Metrics.csv'), encoding='utf-8-sig')
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, all_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title(f'Pipeline 1D-CNN Confusion Matrix (Acc: {final_acc:.2%})', fontsize=16)
    plt.savefig(os.path.join(SAVE_DIR, 'CNN1D_Confusion_Matrix.png'), dpi=150)
    print(f"✅ 结果已保存至 {SAVE_DIR}")


if __name__ == "__main__":
    main()