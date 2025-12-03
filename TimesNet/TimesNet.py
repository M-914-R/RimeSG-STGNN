# model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import pandas as pd


# ==============================
# 2D TimesBlock（自包含，无外部依赖）
# 核心思路：FFT找top-k周期 → [B,C,L]->按周期reshape成2D → Conv2d提取 → 聚合
# ==============================
class TimesBlock2D(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int, top_k: int = 2, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.top_k = top_k
        self.num_layers = num_layers

        blocks = []
        for _ in range(num_layers):
            blocks += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1), groups=hidden_dim),  # 深度可分离(时间方向)
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0), groups=hidden_dim),  # 深度可分离(周期方向)
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Dropout(dropout),
            ]
        self.conv = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _fft_topk_period(x: torch.Tensor, k: int):
        """
        x: [B, L, C]  (这里的C就是hidden_dim)
        返回：
          periods: np.ndarray 形如 [k]，全batch共享的top-k周期
          weights: [B, k]      各batch对这k个频率的权重（幅值均值）
        """
        # 频域: [B, F, C]，F≈L/2+1
        xf = torch.fft.rfft(x, dim=1)
        # 全局（跨batch/通道）求平均幅值，找top-k频点（跳过0频）
        freq_global = xf.abs().mean(dim=0).mean(dim=-1)  # [F]
        if freq_global.numel() > 0:
            freq_global = freq_global.clone()
            freq_global[0] = 0  # 去掉直流分量
        topk_vals, topk_idx = torch.topk(freq_global, k=min(k, freq_global.shape[0]))
        topk_idx = topk_idx.detach().cpu().numpy()  # [k]

        # 对每个batch取对应频率的幅值均值作为聚合权重 [B,k]
        weights = xf.abs().mean(dim=-1)[:, topk_idx]  # [B,k]
        # 周期 = L / 频率索引（避免除0，上面已去0）
        L = x.shape[1]
        periods = np.maximum(1, (L // np.maximum(1, topk_idx)))  # [k]
        return periods, weights

    def forward(self, x: torch.Tensor, extend_len: int):
        """
        x: [B, L, H]  （H=hidden_dim）
        extend_len: 需要扩展到的总长度 L+pred_len
        return: y: [B, L+pred_len, H]
        """
        B, L, H = x.shape
        assert H == self.hidden_dim

        # 若需要在时间维上拼接padding以便卷积输出覆盖 L+pred_len
        if extend_len < L:
            raise ValueError(f"extend_len({extend_len}) must be >= L({L})")
        if extend_len > L:
            pad = torch.zeros(B, extend_len - L, H, device=x.device, dtype=x.dtype)
            x_ext = torch.cat([x, pad], dim=1)  # [B, L+pred_len, H]
        else:
            x_ext = x

        # FFT 选取全局top-k周期 + 权重
        periods, weights = self._fft_topk_period(x_ext, self.top_k)  # periods: [k], weights: [B,k]
        k = len(periods)
        if k == 0:
            # Fallback：若频域异常，直接恒等
            return x_ext

        # 归一化权重做softmax
        weights = F.softmax(weights, dim=1)  # [B,k]

        # 对每个period分别处理，再权重聚合
        agg = 0.0
        for i in range(k):
            p = int(periods[i])
            # 计算能整除的长度
            if x_ext.shape[1] % p != 0:
                length = (x_ext.shape[1] // p + 1) * p
                pad2 = torch.zeros(B, length - x_ext.shape[1], H, device=x.device, dtype=x.dtype)
                xt = torch.cat([x_ext, pad2], dim=1)  # [B, L', H]
            else:
                length = x_ext.shape[1]
                xt = x_ext

            # [B, L', H] → [B, H, L'/p, p]
            xt = xt.view(B, length // p, p, H).permute(0, 3, 1, 2).contiguous()
            yt = self.conv(xt)  # [B, H, L'/p, p]
            # 回到 [B, L', H]
            yt = yt.permute(0, 2, 3, 1).contiguous().view(B, length, H)
            # 截取前 L+pred_len
            yt = yt[:, :x_ext.shape[1], :]  # [B, L+pred_len, H]
            # 按权重聚合
            w = weights[:, i].view(B, 1, 1)
            agg = agg + yt * w

        # 残差 + 归一化
        out = self.ln(agg + x_ext)
        return out  # [B, L+pred_len, H]


# ==============================
# HSTGNN：TimesNet2D 风格（时序） + 仅输出 DO 节点
# 接口与原GWN版保持一致：forward(x:[B,L,N,1]) → [B,1,1,out_len]
# ==============================
class HSTGNN(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        input_dim: int = 1,
        num_nodes: int = 8,
        input_len: int = 12,
        hidden_dim: int = 64,
        output_len: int = 12,
        dropout: float = 0.1,
        adj_path: str = None,
        do_col_name: str = "DO_mgl",
        top_k: int = 2,
        e_layers: int = 2
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.output_len = output_len

        # ===== 解析 DO 列索引（可从邻接CSV列名中提取；否则默认第5列）=====
        self.do_index = 4
        if adj_path is not None and os.path.exists(adj_path):
            try:
                adj_df = pd.read_csv(adj_path, index_col=0)
                cols = list(adj_df.columns)
                if do_col_name in cols:
                    self.do_index = cols.index(do_col_name)
            except Exception:
                pass  # 失败就用默认 4

        # ===== 通道映射：N → H → N =====
        # 我们把“节点数N”当作 TimesNet 的“通道数”
        self.in_proj = nn.Linear(num_nodes, hidden_dim)
        self.blocks = nn.ModuleList([
            TimesBlock2D(hidden_dim=hidden_dim, pred_len=output_len, top_k=top_k, num_layers=1, dropout=dropout)
            for _ in range(e_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

        # dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        """
        x: [B, L, N, 1]
        return: [B, 1, 1, output_len]  —— 仅 DO 节点
        """
        B, L, N, Fin = x.shape
        if Fin != 1:
            raise RuntimeError(f"Expect last dim=1, but got {Fin}")
        if N != self.num_nodes:
            raise RuntimeError(f"N={N} but model.num_nodes={self.num_nodes}")

        # 去掉最后一维 → [B,L,N]
        x = x.squeeze(-1)

        # 简单去趋势标准化（可选）：按时间维做均值方差
        mean = x.mean(dim=1, keepdim=True)              # [B,1,N]
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mean) / std                       # [B,L,N]

        # N → H
        x_h = self.in_proj(x_norm)                      # [B,L,H]

        # TimesNet2D 堆叠，每层都会把时间维扩展成 L+out_len（第一次扩展，后续保持相同长度）
        target_len = L + self.output_len
        for blk in self.blocks:
            x_h = blk(x_h, extend_len=target_len)       # [B, L+out_len, H]
            x_h = self.drop(x_h)

        # H → N
        x_n = self.out_proj(x_h)                        # [B, L+out_len, N]

        # 反标准化
        mean_ext = mean.expand(B, x_n.shape[1], N)      # [B, L+out_len, N]
        std_ext  = std.expand(B, x_n.shape[1], N)
        y = x_n * std_ext + mean_ext                    # [B, L+out_len, N]

        # 只取最后 output_len 个时间步，且只取 DO 节点
        do_idx = min(self.do_index, N - 1)
        y_do = y[:, -self.output_len:, do_idx]          # [B, out_len]

        # 匹配你训练脚本的取出方式（[:,0,0,:]）
        return y_do.unsqueeze(1).unsqueeze(1)           # [B, 1, 1, out_len]
