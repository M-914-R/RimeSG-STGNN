# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# ==============================
# 邻接矩阵规范化（用于 Pearson 图）
# ==============================
def normalize_adj(adj_np: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """
    简单的对称归一化邻接矩阵：D^{-1/2} A D^{-1/2}
    """
    A = np.array(adj_np, dtype=np.float32)
    # 对称化
    A = 0.5 * (A + A.T)
    if add_self_loop:
        np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1) + 1e-6
    d_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


# ==============================
# TimesBlock2D（主时序编码，不带图）
# ==============================
class TimesBlock2D(nn.Module):
    """
    标准 TimesNet 2D Block：
      1) FFT 找 top-k 周期
      2) 按周期 reshape 成 2D
      3) 2D 卷积
      4) 残差 + LayerNorm
    """
    def __init__(self, hidden_dim, pred_len, top_k=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.top_k = top_k

        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3),
                          padding=(0, 1), groups=hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1),
                          padding=(1, 0), groups=hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Dropout(dropout)
            ]
        self.conv = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _fft_topk_period(x, k):
        """
        x: [B, L, H]
        根据频域能量选 top-k 周期
        """
        xf = torch.fft.rfft(x, dim=1)                 # [B, L_f, H]
        freq_global = xf.abs().mean(dim=0).mean(dim=-1)  # [L_f]

        # 去掉直流分量
        if freq_global.numel() > 0:
            freq_global[0] = 0.0

        # 选 top-k 频率
        _, topk_idx = torch.topk(freq_global, k=min(k, freq_global.shape[0]))
        topk_idx = topk_idx.detach().cpu().numpy()

        L = x.shape[1]
        # 频率 -> 周期
        periods = np.maximum(1, (L // np.maximum(1, topk_idx)))

        # 每个 batch 的权重（按幅值统计）
        weights = xf.abs().mean(dim=-1)[:, topk_idx]  # [B, k]

        return periods, weights

    def forward(self, x, extend_len):
        """
        x: [B, L, H]
        """
        B, L, H = x.shape

        # padding 到 extend_len
        if extend_len > L:
            pad = torch.zeros(B, extend_len - L, H,
                              device=x.device, dtype=x.dtype)
            x_ext = torch.cat([x, pad], dim=1)        # [B, extend_len, H]
        else:
            x_ext = x

        # FFT 选周期
        periods, weights = self._fft_topk_period(x_ext, self.top_k)
        if len(periods) == 0:
            return x_ext

        weights = F.softmax(weights, dim=1)           # [B, k]
        agg = 0.0
        length = x_ext.shape[1]

        # 对每个周期做 2D 卷积
        for i, p_i in enumerate(periods):
            p = int(p_i)
            if length % p != 0:
                new_len = (length // p + 1) * p
                pad2 = torch.zeros(B, new_len - length, H,
                                   device=x.device, dtype=x.dtype)
                xt = torch.cat([x_ext, pad2], dim=1)
            else:
                xt = x_ext

            # [B, new_len, H] -> [B, new_len//p, p, H] -> [B, H, new_len//p, p]
            xt = xt.view(B, -1, p, H).permute(0, 3, 1, 2)
            yt = self.conv(xt)
            # -> [B, new_len, H] 再截回原长度
            yt = yt.permute(0, 2, 3, 1).reshape(B, -1, H)[:, :length, :]

            agg = agg + yt * weights[:, i].view(B, 1, 1)

        return self.ln(agg + x_ext)


# ==============================
# 自适应动态图 + Pearson 图 soft-fusion + 空间传播
# ==============================
class GatedDynamicFusion(nn.Module):
    """
    空间模块：
      - Pearson 固定图 A_pearson（由训练集计算的相关图）
      - 自适应图 A_dyn（由可学习嵌入 E_dyn 生成）
      - soft-fusion: A_base = σ(α)*A_pearson + (1 - σ(α))*A_dyn
      - γ(t) 门控对行进行缩放
      - H→N→图卷积→H 的真实空间传播
      - global_w 通道调制
      - reg_term：用于在 loss 中加正则（train.py 已用）
    """
    def __init__(self, num_nodes, hidden_dim, tau_init=1.0, l2_lambda=1e-4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.l2_lambda = l2_lambda

        # 自适应图的节点嵌入
        self.E_dyn = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.1)

        # soft-fusion 权重 α（通过 sigmoid 压到 0~1）
        self.alpha_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # γ(t) 门控（按样本 & 节点）
        self.gate_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_nodes),
            nn.Sigmoid()
        )

        # H -> N -> 图卷积 -> H
        self.h2n = nn.Linear(hidden_dim, num_nodes)   # H → N
        self.n2h = nn.Linear(num_nodes, hidden_dim)   # N → H

        # 温度参数（用于 softmax）
        self.tau_raw = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))

        self.ln = nn.LayerNorm(hidden_dim)

        # 正则项（对图和嵌入做 L2）
        self.reg_term = torch.tensor(0.0)
        # 便于调试可视化
        self.A_last = None

    def forward(self, h_time, A_pearson):
        """
        h_time:   [B, T, H]
        A_pearson: [N, N]  Pearson 固定图（已归一化）
        return:   [B, T, H]
        """
        B, T, H = h_time.shape
        N = self.num_nodes

        tau = F.softplus(self.tau_raw) + 1e-6
        alpha = torch.sigmoid(self.alpha_raw)  # ∈(0,1)

        # ===== 1. 自适应图 A_dyn =====
        h_mean = h_time.mean(dim=(0, 1))                 # [H]
        E_cur = F.normalize(self.E_dyn + h_mean.unsqueeze(0), dim=-1)  # [N, H]

        A_dyn = F.relu(E_cur @ E_cur.T)                  # [N, N]
        A_dyn = A_dyn / (A_dyn.sum(dim=-1, keepdim=True) + 1e-6)
        A_dyn = 0.5 * (A_dyn + A_dyn.T)

        # ===== 2. Pearson + 自适应 soft-fusion =====
        # A_pearson: [N,N]
        A_base = alpha * A_pearson + (1.0 - alpha) * A_dyn  # [N,N]

        # ===== 3. γ(t) 门控：按行缩放（节点重要性）=====
        h_gate = h_time.mean(dim=1)                    # [B, H]
        gamma = self.gate_gen(h_gate)                  # [B, N]

        A_mix = torch.zeros_like(A_base)
        for b in range(B):
            g = gamma[b].view(N, 1)                    # [N,1]
            A_mix_b = A_base * g                       # 行缩放
            A_mix += A_mix_b
        A_mix = A_mix / B                              # [N,N]

        # ===== 4. 图空间传播：H -> N -> A_mix -> H =====
        z_nodes = self.h2n(h_time)                     # [B,T,N]
        z_gcn = torch.matmul(z_nodes, A_mix.T)         # [B,T,N]
        msg_h = self.n2h(z_gcn)                        # [B,T,H]

        # ===== 5. 通道级调制（global_w）=====
        logits = (A_base @ E_cur) / tau                # [N,H]
        node_attn = torch.softmax(logits, dim=0)       # [N,H]
        global_w = node_attn.mean(dim=0, keepdim=True) # [1,H]
        global_w = torch.tanh(global_w)                # [-1,1]

        # ===== 6. 融合：原特征 * 通道调制 + 图消息残差 =====
        y = h_time * (1.0 + global_w) + msg_h          # [B,T,H]
        y = self.ln(y)

        # ===== 7. 正则项（给 loss 用）=====
        self.reg_term = self.l2_lambda * (
            A_dyn.pow(2).mean() + E_cur.pow(2).mean()
        )
        self.A_last = A_mix.detach()

        return y


# ==============================
# HSTGNN（TimesNet ↔ 图模块交替 + Pearson+自适应 soft-fusion）
# ==============================
class HSTGNN(nn.Module):
    def __init__(
        self,
        device=torch.device("cpu"),
        input_dim=1,
        num_nodes=8,
        input_len=12,
        hidden_dim=64,
        output_len=1,
        dropout=0.1,
        adj_path="./data/adj_matrices/cleaned_dataset1_adj.csv",  # Pearson 图
        do_col_name="DO_mgl",
        top_k=2,
        e_layers=2
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_len = output_len

        # ===== 1. 读取 Pearson 图，归一化后作为 A_pearson =====
        if (adj_path is None) or (not os.path.exists(adj_path)):
            # 找不到就用单位阵
            A_np = np.eye(num_nodes, dtype=np.float32)
            col_names = [f"node_{i}" for i in range(num_nodes)]
        else:
            adj_df = pd.read_csv(adj_path, index_col=0)
            col_names = list(adj_df.columns)
            A_np = adj_df.to_numpy().astype(np.float32)

        A_norm = normalize_adj(A_np, add_self_loop=True)
        self.register_buffer("A_pearson", torch.tensor(A_norm, dtype=torch.float32))

        # ===== 2. DO 节点索引（用于最后只取 DO_mgl）=====
        self.do_index = 4
        if do_col_name in col_names:
            self.do_index = col_names.index(do_col_name)

        # ===== 3. 时序编码（TimesNet 主干）=====
        self.in_proj = nn.Linear(num_nodes, hidden_dim)
        self.times_blocks = nn.ModuleList([
            TimesBlock2D(hidden_dim, output_len, top_k=top_k,
                         num_layers=1, dropout=dropout)
            for _ in range(e_layers)
        ])

        self.graph_fusion = GatedDynamicFusion(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            tau_init=1.0,
            l2_lambda=1e-4
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x):
        """
        x: [B, L, N, 1]
        返回: [B, 1, 1, output_len]
        """
        B, L, N, Fin = x.shape
        assert Fin == 1 and N == self.num_nodes

        # ===== 标准化（逐样本）=====
        x = x.squeeze(-1)                             # [B,L,N]
        mean = x.mean(dim=1, keepdim=True)            # [B,1,N]
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6)                  # [B,1,N]
        x_norm = (x - mean) / std                     # [B,L,N]

        # ===== 输入映射到隐藏维 =====
        h = self.in_proj(x_norm)                      # [B,L,H]
        target_len = L + self.output_len
        A_pearson = self.A_pearson.to(h.device)       # [N,N]

        # ===== 时-空交替堆叠：TimesNet → GraphFusion → ... =====
        for blk in self.times_blocks:
            # 时间编码（周期抽取 + 2D 卷积）
            h = blk(h, extend_len=target_len)         # [B,target_len,H]
            # 空间编码（Pearson + 自适应图 soft-fusion）
            h = self.graph_fusion(h, A_pearson)       # [B,target_len,H]
            h = self.dropout(h)

        # ===== 映射回节点维度 =====
        y_nodes = self.out_proj(h)                    # [B,target_len,N]

        # ===== 反标准化 =====
        mean_ext = mean.expand(B, y_nodes.shape[1], N)
        std_ext = std.expand(B, y_nodes.shape[1], N)
        y_nodes = y_nodes * std_ext + mean_ext        # [B,target_len,N]

        # ===== 只取 DO 节点，最后 output_len 个时间步 =====
        do_idx = min(self.do_index, N - 1)
        y_do = y_nodes[:, -self.output_len:, do_idx]  # [B,output_len]

        # 保持和 train.py 一致的输出形状
        return y_do.unsqueeze(1).unsqueeze(1)         # [B,1,1,output_len]
