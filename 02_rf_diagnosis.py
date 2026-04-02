import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DENOISED_DIR = './Pipeline_Denoised_Data'
SAVE_DIR = './Pipeline_RF_Results'
os.makedirs(SAVE_DIR, exist_ok=True)
CLASSES = ['0.4mm leak', '2mm leak', '4mm leak', 'no Leak']
LABEL_MAP = {CLASSES[0]: 0, CLASSES[1]: 1, CLASSES[2]: 2, CLASSES[3]: 3}
CHUNK_SIZE = 1024


# ================= 纯手写决策树与随机森林 =================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature;
        self.threshold = threshold
        self.left = left;
        self.right = right;
        self.value = value

    def is_leaf_node(self): return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split;
        self.max_depth = max_depth;
        self.n_features = n_features;
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split):
            return Node(value=Counter(y).most_common(1)[0][0])
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thr = self._best_split(X, y, feat_idxs)
        if best_feat is None: return Node(value=Counter(y).most_common(1)[0][0])
        l_idxs, r_idxs = np.argwhere(X[:, best_feat] <= best_thr).flatten(), np.argwhere(
            X[:, best_feat] > best_thr).flatten()
        return Node(best_feat, best_thr, self._grow_tree(X[l_idxs], y[l_idxs], depth + 1),
                    self._grow_tree(X[r_idxs], y[r_idxs], depth + 1))

    def _best_split(self, X, y, feat_idxs):
        best_gain, s_idx, s_thr = -1, None, None
        for feat in feat_idxs:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) > 15: thresholds = np.percentile(col, np.linspace(5, 95, 15))
            for thr in thresholds:
                l_idxs, r_idxs = np.argwhere(col <= thr).flatten(), np.argwhere(col > thr).flatten()
                if len(l_idxs) == 0 or len(r_idxs) == 0: continue
                n = len(y)
                _, cl = np.unique(y[l_idxs], return_counts=True);
                el = 1.0 - sum((cl / cl.sum()) ** 2)
                _, cr = np.unique(y[r_idxs], return_counts=True);
                er = 1.0 - sum((cr / cr.sum()) ** 2)
                _, cp = np.unique(y, return_counts=True);
                p_gini = 1.0 - sum((cp / cp.sum()) ** 2)
                gain = p_gini - ((len(l_idxs) / n) * el + (len(r_idxs) / n) * er)
                if gain > best_gain: best_gain, s_idx, s_thr = gain, feat, thr
        return s_idx, s_thr

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf_node(): return node.value
        return self._traverse(x, node.left) if x[node.feature] <= node.threshold else self._traverse(x, node.right)


class CustomRandomForest:
    def __init__(self, n_trees=50, max_depth=15):
        self.n_trees = n_trees;
        self.max_depth = max_depth;
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, n_features=int(np.sqrt(X.shape[1])))
            idxs = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(p).most_common(1)[0][0] for p in np.swapaxes(preds, 0, 1)])


# ================= 主流程 =================
def main():
    print("1. 正在读取清洗后的数据，并提取时域统计特征...")
    features, labels = [], []
    for c_name in CLASSES:
        files = glob.glob(os.path.join(DENOISED_DIR, c_name, '*.csv'))
        for f in files:
            signal = pd.read_csv(f, header=None).values.flatten()
            for j in range(len(signal) // CHUNK_SIZE):
                chunk = signal[j * CHUNK_SIZE: (j + 1) * CHUNK_SIZE]
                rms = np.sqrt(np.mean(chunk ** 2))
                features.append([np.mean(chunk), np.std(chunk), rms, np.max(np.abs(chunk)),
                                 skew(chunk), kurtosis(chunk), np.max(np.abs(chunk)) / (rms + 1e-8),
                                 rms / (np.mean(np.abs(chunk)) + 1e-8)])
                labels.append(LABEL_MAP[c_name])

    X, y = np.array(features), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("2. 正在训练自定义随机森林 (100棵树)...")
    rf = CustomRandomForest(n_trees=100)
    start_t = time.time()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 自研 RF 准确率: {acc:.2%} (耗时: {time.time() - start_t:.2f}秒)")

    pd.DataFrame(classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)).transpose().to_csv(
        os.path.join(SAVE_DIR, 'RF_Metrics.csv'), encoding='utf-8-sig')

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title(f'Pipeline Custom RF Confusion Matrix (Acc: {acc:.2%})', fontsize=16)
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_DIR, 'RF_Confusion_Matrix.png'), dpi=150)
    print(f"✅ 结果已保存至 {SAVE_DIR}")


if __name__ == "__main__":
    main()