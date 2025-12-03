# -*- coding: utf-8 -*-
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")      # ← 放在 import matplotlib.pyplot 之前
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from model import HSTGNN   #  模型外置
from ranger21 import Ranger


# ============================================================
# 参数设置
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="")
parser.add_argument("--data", type=str, default="cleaned_dataset3", help="dataset name (without extension)")
parser.add_argument("--input_dim", type=int, default=1, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="hidden dimension")
parser.add_argument("--num_nodes", type=int, default=8, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=1, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--epochs", type=int, default=500, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--save", type=str, default="./log/" + str(time.strftime("%Y-%m-%d-%H_%M_%S")) + "-",
                    help="save path")
parser.add_argument("--es_patience", type=int, default=50,
                    help="early stop patience")
args = parser.parse_args()


# === 动态生成 save 路径 ===
timestamp = time.strftime("%Y-%m-%d-%H_%M_%S")
args.save = f"./log/{timestamp}-{args.data}-out{args.output_len}/"


# ============================================================
# 固定随机种子
# ============================================================
def seed_it(seed=42):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


# ============================================================
# Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, save_path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


# ============================================================
# 数据加载（StandardScaler + SG滤波 + 输入输出分开标准化）
# ============================================================
# ============================================================
# 数据加载
# ============================================================
def create_inout_sequences_xy(data_X, data_Y, input_len, output_len):
    X, Y = [], []
    for i in range(len(data_X) - input_len - output_len):
        X.append(data_X[i:i + input_len])                  # 输入
        Y.append(data_Y[i + input_len:i + input_len + output_len])  # 标签（原始 DO）
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def load_dataset(file_path, input_len, output_len, batch_size,
                 train_ratio=0.8, val_ratio=0.1):

    from scipy.signal import savgol_filter
    from sklearn.preprocessing import StandardScaler

    print(f" 加载数据文件: {file_path}")

    # ===== 1) 读数据 =====
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    data_raw = df[num_cols].values.astype(np.float32)   # 原始数据
    data_X = data_raw.copy()                            # 输入用（SG）
    data_Y = data_raw[:, [num_cols.index("DO_mgl")]]    # 标签只取 DO 列（原始）

    # ===== 2) 对输入做 SG 滤波，仅 DO 列 =====
    do_idx = num_cols.index("DO_mgl")
    do_raw = data_raw[:, do_idx]
    window_length = 11 if len(data_raw) > 11 else (len(data_raw)//2*2-1)
    do_sg = savgol_filter(do_raw, window_length=window_length, polyorder=3)

    data_X[:, do_idx] = do_sg.astype(np.float32)   # 输入 DO ← SG 去噪
    # data_Y 保持完全原始 DO

    # ===== 3) 划分 train/val/test，不泄露 =====
    n_total = len(data_raw)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * (train_ratio + val_ratio))

    X_train_raw, X_val_raw, X_test_raw = data_X[:n_train], data_X[n_train:n_val], data_X[n_val:]
    Y_train_raw, Y_val_raw, Y_test_raw = data_Y[:n_train], data_Y[n_train:n_val], data_Y[n_val:]

    # ===== 4) 为 X 和 Y 分别创建 scaler（非常重要！）=====
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_val   = X_scaler.transform(X_val_raw)
    X_test  = X_scaler.transform(X_test_raw)

    Y_train = Y_scaler.fit_transform(Y_train_raw)
    Y_val   = Y_scaler.transform(Y_val_raw)
    Y_test  = Y_scaler.transform(Y_test_raw)

    # ===== 5) 滑窗 =====
    X_train_seq, Y_train_seq = create_inout_sequences_xy(X_train, Y_train, input_len, output_len)
    X_val_seq,   Y_val_seq   = create_inout_sequences_xy(X_val,   Y_val,   input_len, output_len)
    X_test_seq,  Y_test_seq  = create_inout_sequences_xy(X_test,  Y_test,  input_len, output_len)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq), torch.tensor(Y_train_seq)),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val_seq), torch.tensor(Y_val_seq)),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test_seq), torch.tensor(Y_test_seq)),
                              batch_size=batch_size, shuffle=False)

    # 返回两个 scaler
    return train_loader, val_loader, test_loader, Y_test, X_scaler, Y_scaler, num_cols



# ============================================================
# 指标函数
# ============================================================
def _flatten_tensors(pred, true, mask_value=None):
    """通用展开与掩码函数"""
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return pred, true


def MAE_torch(pred, true, mask_value=None):
    pred, true = _flatten_tensors(pred, true, mask_value)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    pred, true = _flatten_tensors(pred, true, mask_value)
    denom = torch.where(torch.abs(true) < 1e-6, torch.full_like(true, 1e-6), true)
    return torch.mean(torch.abs((true - pred) / denom))  # 转为百分比


def RMSE_torch(pred, true, mask_value=None):
    pred, true = _flatten_tensors(pred, true, mask_value)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    pred, true = _flatten_tensors(pred, true, mask_value)
    denom = torch.sum(torch.abs(true)) + 1e-6
    return torch.sum(torch.abs(pred - true)) / denom


def R2_torch(pred, true, mask_value=None):
    pred, true = _flatten_tensors(pred, true, mask_value)
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2) + 1e-6
    return 1 - ss_res / ss_tot


def metric(pred, true):
    """统一输出五个指标"""
    mae = MAE_torch(pred, true, 0.0).item()
    mape = MAPE_torch(pred, true, 0.0).item()
    rmse = RMSE_torch(pred, true, 0.0).item()
    wmape = WMAPE_torch(pred, true, 0.0).item()
    r2 = R2_torch(pred, true, 0.0).item()
    return mae, mape, rmse, wmape, r2



# ============================================================
# 主函数 for Koopman model
# ============================================================
def main():
    seed_it(6666)
    device = torch.device(args.device)

    # ===== 1. 数据路径与 DO 列 =====
    data_path = f"data/cleaned/{args.data}.xlsx"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" 找不到数据文件: {data_path}")

    df = pd.read_excel(data_path)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    print(f" 数值列: {num_cols}")

    DO_COL = "DO_mgl"
    do_idx = 4  # 第5列
    print(f" 目标列: {DO_COL} (索引 {do_idx})")

    # ===== 2. 数据加载 =====
    train_loader, val_loader, test_loader, Y_test_raw, X_scaler, Y_scaler, num_cols = load_dataset(
        data_path, args.input_len, args.output_len, args.batch_size
    )

    # 用全量特征做一个 StandardScaler（用于 inverse_only_do）
    mm = StandardScaler()
    mm.fit(df[num_cols].values.astype(np.float32))

    # ===== 3. 模型初始化 =====
    model = HSTGNN(
        device=device,
        input_dim=1,
        num_nodes=8,
        input_len=args.input_len,
        hidden_dim=64,
        output_len=args.output_len,
        dropout=0.1,
        adj_path="./data/adj_matrices/cleaned_dataset3_adj.csv",
        do_col_name="DO_mgl",
        top_k=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #optimizer = Ranger(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(
        patience=args.es_patience,
        verbose=True,
        save_path=os.path.join(args.save, args.data, "best_model.pth")
    )

    os.makedirs(os.path.join(args.save, args.data), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_epoch_loss, valid_epoch_loss = [], []
    train_time, val_time = [], []
    his_loss = []
    result, test_result = [], []

    print("start training...")

    # ===== 4. 训练循环 =====
    for epoch in range(1, args.epochs + 1):
        model.train()
        t1 = time.time()
        train_loss, train_mape, train_rmse, train_wmape, train_r2 = [], [], [], [], []

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            # === 模型期望输入 [B, L, N, 1] ===
            if X.ndim == 3:  # 如果 loader 给的是 [B, L, N]
                X = X.unsqueeze(-1)

            optimizer.zero_grad()

            # === 前向传播 ===
            pred = model(X)[:, 0, 0, :]  # [B, output_len]
            # === Ground truth ===
            # 若 load_dataset 输出 [B, output_len, N]：
            true = Y  # [B, output_len]

            # === 损失 ===
            loss = MAE_torch(pred, true)
            loss += 1e-4 * model.graph_fusion.reg_term



            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(MAPE_torch(pred, true).item())
            train_rmse.append(RMSE_torch(pred, true).item())
            train_wmape.append(WMAPE_torch(pred, true).item())
            train_r2.append(R2_torch(pred, true).item())

        t2 = time.time()
        train_time.append(t2 - t1)
        mtrain_loss, mtrain_mape = np.mean(train_loss), np.mean(train_mape)
        mtrain_rmse, mtrain_wmape, mtrain_r2 = np.mean(train_rmse), np.mean(train_wmape), np.mean(train_r2)

        # ===== 验证 =====
        model.eval()
        s1 = time.time()
        val_loss, val_mape, val_rmse, val_wmape, val_r2 = [], [], [], [], []

        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                if Xv.ndim == 3:
                    Xv = Xv.unsqueeze(-1)

                pred = model(Xv)[:, 0, 0, :]  # [B, output_len]
                true = Yv  # [B, output_len]

                val_loss.append(MAE_torch(pred, true).item())
                val_mape.append(MAPE_torch(pred, true).item())
                val_rmse.append(RMSE_torch(pred, true).item())
                val_wmape.append(WMAPE_torch(pred, true).item())
                val_r2.append(R2_torch(pred, true).item())

        s2 = time.time()
        val_time.append(s2 - s1)
        mvalid_loss, mvalid_mape = np.mean(val_loss), np.mean(val_mape)
        mvalid_rmse, mvalid_wmape, mvalid_r2 = np.mean(val_rmse), np.mean(val_wmape), np.mean(val_r2)

        print(f"Epoch: {epoch:03d}, Train Loss: {mtrain_loss:.4f}, RMSE: {mtrain_rmse:.4f}, "
              f"MAPE: {mtrain_mape:.4f}, WMAPE: {mtrain_wmape:.4f}, R2: {mtrain_r2:.4f}")
        print(f"Epoch: {epoch:03d}, Valid Loss: {mvalid_loss:.4f}, RMSE: {mvalid_rmse:.4f}, "
              f"MAPE: {mvalid_mape:.4f}, WMAPE: {mvalid_wmape:.4f}, R2: {mvalid_r2:.4f}")

        train_epoch_loss.append(mtrain_loss)
        valid_epoch_loss.append(mvalid_loss)
        his_loss.append(mvalid_loss)

        metrics_dict = dict(
            train_loss=mtrain_loss, train_rmse=mtrain_rmse, train_mape=mtrain_mape,
            train_wmape=mtrain_wmape, train_r2=mtrain_r2,
            valid_loss=mvalid_loss, valid_rmse=mvalid_rmse,
            valid_mape=mvalid_mape, valid_wmape=mvalid_wmape, valid_r2=mvalid_r2
        )
        result.append(pd.Series(metrics_dict))

        early_stopping(mvalid_loss, model)
        if early_stopping.early_stop and epoch >= 50:
            print(f"Early stopping at epoch {epoch}")
            break

        pd.DataFrame(result).round(8).to_csv(f"{args.save}/{args.data}/train.csv")

        # ============================================================
        # 5. 测试阶段（支持 1、3、6、9 步预测）
        # ============================================================
    model.load_state_dict(torch.load(os.path.join(args.save, args.data, "best_model.pth"), map_location=device))
    model.eval()

    outputs_scaled, reals_scaled = [], []
    with torch.no_grad():
        for Xt, Yt in test_loader:
            Xt, Yt = Xt.to(device), Yt.to(device)
            if Xt.ndim == 3:
                Xt = Xt.unsqueeze(-1)
            pred = model(Xt)[:, 0, 0, :]  # [B, output_len]
            outputs_scaled.append(pred.cpu())
            reals_scaled.append(Yt.cpu())

    yhat_scaled = torch.cat(outputs_scaled, dim=0)  # [T, output_len]
    ytrue_scaled = torch.cat(reals_scaled, dim=0)  # [T, output_len]

    # ===== 反归一化（仅 DO 列）=====
    def inverse_only_do(x_scaled):
        """
        x_scaled: torch.Tensor of shape [T, output_len]
        使用标签的 scaler (Y_scaler) 反标准化 DO
        """
        x_np = x_scaled.cpu().numpy().reshape(-1, 1)  # [T*out_len, 1]
        x_inv = Y_scaler.inverse_transform(x_np)  # 反标准化到真实 DO
        return torch.from_numpy(x_inv.reshape(x_scaled.shape))

    yhat = inverse_only_do(yhat_scaled)
    ytrue = inverse_only_do(ytrue_scaled)

    # ====== 保存真实值与预测值（默认保存第1步）======
    df_out = pd.DataFrame({
        "ytrue": ytrue.reshape(-1).cpu().numpy(),
        "yhat": yhat.reshape(-1).cpu().numpy(),
    })

    output_dir = os.path.join(args.save, args.data)
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "pred_real_DO.csv")
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"✅ 测试集真实值与预测值已保存到：{save_path}")

    # ============================================================
    # 6. 多步预测指标计算与保存（1~12步单点指标 + 前1/3/6/9/12步平均指标）
    # ============================================================

    point_horizons = list(range(1, 13))  # 单点指标: 第1~12步
    cumul_horizons = [1, 3, 6, 9, 12]  # 前N步累计指标

    combined_metrics = []

    print("\n========== Multi-Horizon Test Results ==========")

    # ===== 自动确定模型的预测步数 =====
    max_horizon = yhat.shape[1]  # 即 output_len
    point_horizons = list(range(1, max_horizon + 1))  # 单点指标
    combined_metrics = []

    # --- 单点指标 ---
    print("\n--- Point-wise Metrics ---")
    for h in point_horizons:
        pred_h = yhat[:, h - 1]
        true_h = ytrue[:, h - 1]
        mae, mape, rmse, wmape, r2 = metric(pred_h, true_h)
        print(f"[Point@{h:02d}]  MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
              f"MAPE: {mape:.4f} | WMAPE: {wmape:.4f} | R²: {r2:.4f}")
        combined_metrics.append({
            "Horizon": h,
            "Type": f"Point@{h}",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "WMAPE": wmape,
            "R2": r2
        })

    # --- 全步平均指标（前 output_len 步） ---
    print("\n--- Average over all horizons ---")
    pred_c = yhat[:, :max_horizon].reshape(-1)
    true_c = ytrue[:, :max_horizon].reshape(-1)
    mae, mape, rmse, wmape, r2 = metric(pred_c, true_c)
    print(f"[Cumul@{max_horizon:02d}] Average over {max_horizon} steps → "
          f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f} | WMAPE: {wmape:.4f} | R²: {r2:.4f}")
    combined_metrics.append({
        "Horizon": max_horizon,
        "Type": f"Cumul@{max_horizon}",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WMAPE": wmape,
        "R2": r2
    })

    # === 保存所有指标到一个文件 ===
    df_metrics = pd.DataFrame(combined_metrics).round(6)
    output_dir = os.path.join(args.save, args.data)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "multi_horizon_metrics_combined.csv")
    df_metrics.to_csv(csv_path, index=False)

    # === 绘制 MAE 对比图 ===
    plt.figure(figsize=(6, 4))
    point_df = df_metrics[df_metrics["Type"].str.contains("Point")]
    plt.plot(point_df["Horizon"], point_df["MAE"], marker='o', label="MAE-Point")
    plt.axhline(y=df_metrics.iloc[-1]["MAE"], color='r', linestyle='--',
                label=f"Avg MAE (1~{max_horizon})={df_metrics.iloc[-1]['MAE']:.4f}")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("MAE")
    plt.title(f"Multi-Horizon MAE (1 ~ {max_horizon} steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_horizon_mae_comparison.png"))
    plt.close()

    print(f"\n✅ 已输出模型 {max_horizon} 步的单点指标与整体平均指标")
    print(f"✅ 指标文件保存至: {csv_path}")
    print(f"✅ MAE 可视化图保存为: {output_dir}/multi_horizon_mae_comparison.png")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

