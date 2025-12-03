# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as np
import pandas as pd

# ===== 路径配置 =====
DATA_DIR = "./data/cleaned"
SAVE_DIR = "./data/adj_matrices"
os.makedirs(SAVE_DIR, exist_ok=True)

# 时间列与特征列（确保与你模型的节点一一对应）
TIME_COLS = {"DateTimeStamp", "datetime", "time", "日期", "时间"}
FEATURE_COLS = ['Temp', 'SpCond', 'Sal', 'DO_pct', 'DO_mgl', 'Depth', 'pH', 'Turb']

# ===== 参数可调 =====
USE_ABS = True       # 是否取绝对值（常开）
ZERO_DIAG = True     # 是否置零对角线
THRESHOLD = 0.3      # 去噪阈值，建议 0.3~0.5
USE_GAUSSIAN = True  # 是否启用高斯核平滑
ADD_SELF_LOOP = True # 是否加自环

# ===== 文件名识别 =====
def _extract_dataset_index(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r'cleaned_dataset\s*(\d+)\.(xlsx|csv)$', base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]


# ===== 主函数：根据 Pearson 生成图 =====
# ===== 主函数：根据 Pearson 生成图（仅使用训练集） =====
def build_from_file(xf: str, train_ratio: float = 0.8):
    if not os.path.exists(xf):
        print(f"[Skip] {xf} 不存在")
        return

    idx = _extract_dataset_index(xf)
    print(f"[Info] Building Pearson-Gaussian adjacency for dataset {idx}: {xf}")

    # 读表
    if xf.lower().endswith(".csv"):
        df = pd.read_csv(xf)
    else:
        df = pd.read_excel(xf)

    # 去掉时间列
    for c in list(df.columns):
        if c in TIME_COLS:
            df = df.drop(columns=[c])

    # 仅保留 8 个指标
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[Dataset {idx}] 缺少必要列: {missing}；现有列: {list(df.columns)}")

    df = df[FEATURE_COLS].astype(np.float32)
    df = df.interpolate(limit_direction='both').bfill().ffill()

    # ===== ✨ 仅用训练集计算相关系数 ✨ =====
    n = len(df)
    n_train = int(n * train_ratio)
    df_train = df.iloc[:n_train]
    print(f"[Train only] 使用前 {n_train}/{n} ({train_ratio*100:.1f}%) 样本计算相关系数")

    # ===== Pearson 相关系数 =====
    pear = df_train.corr(method='pearson').to_numpy().astype(np.float32)
    pear = np.nan_to_num(pear, nan=0.0)
    if USE_ABS:
        pear = np.abs(pear)
    if ZERO_DIAG:
        np.fill_diagonal(pear, 0.0)

    # ===== 阈值筛选 =====
    pear[pear < THRESHOLD] = 0.0

    # ===== 高斯核平滑 =====
    if USE_GAUSSIAN:
        dist = 1 - pear
        sigma = np.std(dist[dist > 0]) if np.any(dist > 0) else 1.0
        pear = np.exp(-dist**2 / (2 * sigma**2))

    # ===== 自环 + 对称化 + 归一化 =====
    if ADD_SELF_LOOP:
        np.fill_diagonal(pear, 1.0)
    pear = 0.5 * (pear + pear.T)
    pear = pear / (pear.max() + 1e-6)  # 缩放到 [0,1]

    # ===== 保存结果 =====
    out_csv = os.path.join(SAVE_DIR, f"cleaned_dataset{idx}_adj.csv")
    pd.DataFrame(pear, index=FEATURE_COLS, columns=FEATURE_COLS)\
        .to_csv(out_csv, float_format="%.6f", encoding="utf-8-sig")

    print(f"[✅ Saved] {out_csv}")
    print(f"shape={pear.shape}, min={pear.min():.3f}, max={pear.max():.3f}, mean={pear.mean():.3f}")
    print(f"THRESHOLD={THRESHOLD}, GAUSSIAN={USE_GAUSSIAN}, SELF_LOOP={ADD_SELF_LOOP}\n")



# ===== 主入口 =====
if __name__ == "__main__":
    pattern = os.path.join(DATA_DIR, "cleaned_dataset*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[Warn] 未在 {DATA_DIR} 发现 cleaned_dataset*.xlsx")
    for xf in files:
        build_from_file(xf)
