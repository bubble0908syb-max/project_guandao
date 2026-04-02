import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from vmdpy import VMD

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = './Pipeline_Data'
OUTPUT_DIR = './Pipeline_Denoised_Data'
PLOT_DIR = './VMD_Plots'
CHUNK_SIZE = 1024
CLASSES = ['0.4mm leak', '2mm leak', '4mm leak', 'no Leak']

for d in [OUTPUT_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)


def denoise_vmd(signal):
    """单帧 VMD 去噪函数"""
    alpha, tau, K, DC, init, tol = 2000, 0, 6, 0, 1, 1e-7
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    correlations = [abs(pearsonr(mode, signal)[0]) for mode in u]
    threshold = np.mean(correlations)

    denoised_signal = np.zeros_like(signal)
    selected = 0
    for i, mode in enumerate(u):
        if correlations[i] >= threshold:
            denoised_signal += mode
            selected += 1

    return denoised_signal if selected > 0 else signal


def main():
    print("================ 开始全量数据 VMD 去噪清洗 ================")
    plot_samples = []  # 用于保存四个类别的对比画图数据

    for class_name in CLASSES:
        in_folder = os.path.join(INPUT_DIR, class_name)
        out_folder = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.exists(in_folder):
            print(f"⚠️ 找不到文件夹: {in_folder}")
            continue

        files = glob.glob(os.path.join(in_folder, '*.*'))
        print(f"\n📁 正在处理 [{class_name}] 类别，共 {len(files)} 个文件...")

        for file_idx, file in enumerate(files):
            # 读取原始数据
            df = pd.read_csv(file, header=None) if file.endswith('.csv') else pd.read_excel(file, header=None,
                                                                                            engine='openpyxl')
            signal = df.values.flatten()

            clean_signal_parts = []
            # 分帧去噪
            for j in range(len(signal) // CHUNK_SIZE):
                chunk = signal[j * CHUNK_SIZE: (j + 1) * CHUNK_SIZE]
                clean_chunk = denoise_vmd(chunk)
                clean_signal_parts.extend(clean_chunk)

                # 收集每个类别的第一帧用于画对比图
                if file_idx == 0 and j == 0 and len(plot_samples) < 4:
                    plot_samples.append((class_name, chunk, clean_chunk))

            # 保存去噪后的数据至新目录
            base_name = os.path.basename(file)
            save_path = os.path.join(out_folder, f"denoised_{base_name.replace('.xlsx', '.csv')}")
            pd.DataFrame(clean_signal_parts).to_csv(save_path, index=False, header=False)

            if (file_idx + 1) % 10 == 0:
                print(f"  -> 已清洗 {file_idx + 1} / {len(files)} 个文件")

    # 绘制并保存去噪效果对比大图
    print("\n📸 正在生成四个类别的去噪效果对比图...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle("管道泄漏不同工况 VMD 去噪效果对比 (单帧1024点)", fontsize=18, fontweight='bold')

    for idx, (c_name, raw_c, clean_c) in enumerate(plot_samples):
        axes[idx, 0].plot(raw_c, color='gray', alpha=0.8)
        axes[idx, 0].set_title(f"[{c_name}] 原始带噪信号")
        axes[idx, 0].set_ylabel("幅值")

        axes[idx, 1].plot(clean_c, color='red', alpha=0.9)
        axes[idx, 1].set_title(f"[{c_name}] VMD 去噪后信号")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'VMD_Denoise_Comparison.png'), dpi=150)
    print("✅ 数据清洗完毕！所有干净文件已存入 Pipeline_Denoised_Data 文件夹。")


if __name__ == "__main__":
    main()