# rime_hpo.py
import numpy as np
import random
from dataclasses import dataclass

@dataclass
class RimeBounds:
    """每个超参数的一维连续或离散取值范围/集合"""
    # 连续区间用 (low, high)，离散集合用 list
    space: dict  # 例：{"lr": (1e-4, 3e-3), "dropout": (0.0, 0.5), "hidden_dim": [64, 96, 128]}

def _sample_from_space(space):
    cfg = {}
    for k, v in space.items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            # 连续
            low, high = v
            cfg[k] = float(np.random.uniform(low, high))
        elif isinstance(v, list):
            # 离散集合
            cfg[k] = random.choice(v)
        else:
            raise ValueError(f"Unsupported space for key={k}: {v}")
    return cfg

def _clip_to_space(cfg, space):
    out = {}
    for k, v in space.items():
        x = cfg[k]
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(a, (int, float)) for a in v):
            low, high = v
            out[k] = float(np.clip(x, low, high))
        elif isinstance(v, list):
            # 离散空间：就近取整 + clamp 后再 map 到集合索引
            arr = v
            # 用最近邻离散化
            idx = np.argmin([abs(x - a if isinstance(a,(int,float)) else 1e9) for a in arr])
            out[k] = arr[idx]
        else:
            out[k] = x
    return out

def _cfg_to_vec(cfg, space):
    vec = []
    meta = []
    for k, v in space.items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(a, (int, float)) for a in v):
            vec.append(float(cfg[k]))
            meta.append(("cont", k, v))
        elif isinstance(v, list):
            arr = v
            # 用值在集合中的索引（若不在集合，取最近）
            if cfg[k] in arr:
                idx = arr.index(cfg[k])
            else:
                idx = int(np.argmin([abs(cfg[k] - a if isinstance(a,(int,float)) else 1e9) for a in arr]))
            vec.append(float(idx))
            meta.append(("disc", k, arr))
        else:
            raise ValueError("Unsupported space")
    return np.array(vec, dtype=np.float32), meta

def _vec_to_cfg(vec, meta):
    cfg = {}
    for i, (tp, key, info) in enumerate(meta):
        if tp == "cont":
            low, high = info
            cfg[key] = float(np.clip(vec[i], low, high))
        else:
            arr = info
            idx = int(np.clip(round(vec[i]), 0, len(arr)-1))
            cfg[key] = arr[idx]
    return cfg

class RimeHPO:
    """
    雾凇优化算法（RIME）用于超参数优化：
    - 种群初始化 → 评估 → 凝结（向全局最优聚集）→ 冻结（保留优秀个体）→ 扩散（维持多样性）
    - 适合和“训练少量 epoch 的验证损失”配合，迭代搜索最优超参
    """
    def __init__(self, bounds: RimeBounds, pop_size=12, iters=10, coalesce=0.8, disperse=0.2, freeze_ratio=0.2, seed=42):
        self.bounds = bounds.space
        self.pop_size = pop_size
        self.iters = iters
        self.coalesce = coalesce    # 凝结强度 beta1
        self.disperse = disperse    # 扩散强度 beta2
        self.freeze_ratio = freeze_ratio
        self.rng = np.random.RandomState(seed)

    def optimize(self, eval_fn, verbose=True):
        """
        eval_fn(cfg) -> fitness(float)
        这里 fitness = 验证集损失（越小越好）
        """
        # 初始化
        pop_cfg = [_sample_from_space(self.bounds) for _ in range(self.pop_size)]
        pop_vec_meta = [_cfg_to_vec(c, self.bounds) for c in pop_cfg]
        pop_vec = np.stack([v for v, _ in pop_vec_meta], axis=0)
        meta = pop_vec_meta[0][1]

        # 初次评估
        fitness = np.array([eval_fn(c) for c in pop_cfg], dtype=np.float32)
        best_idx = int(np.argmin(fitness))
        best_vec = pop_vec[best_idx].copy()
        best_cfg = pop_cfg[best_idx]
        best_fit = float(fitness[best_idx])

        if verbose:
            print(f"[RIME] init best: {best_cfg}  val_loss={best_fit:.6f}")

        for t in range(self.iters):
            # 冻结（保留前 freeze_ratio 的个体）
            k_freeze = max(1, int(self.freeze_ratio * self.pop_size))
            elite_idx = np.argsort(fitness)[:k_freeze]
            elites = pop_vec[elite_idx].copy()

            new_pop = []
            for i in range(self.pop_size):
                xi = pop_vec[i].copy()
                # 随机挑选一个个体用于扩散项
                j = self.rng.randint(0, self.pop_size)
                while j == i:
                    j = self.rng.randint(0, self.pop_size)
                xj = pop_vec[j]

                # 凝结 + 扩散（核心雾凇更新）
                # Xi_{t+1} = Xi_t + beta1 * (Xbest - |Xi_t|) + beta2 * (Xrand - Xi_t) + noise
                noise_scale = max(1e-3, 0.1 * (1 - t / max(1, self.iters-1)))  # 逐步衰减的噪声
                xi_new = (xi
                          + self.coalesce * (best_vec - np.abs(xi))
                          + self.disperse * (xj - xi)
                          + self.rng.normal(0.0, noise_scale, size=xi.shape))

                new_pop.append(xi_new)

            # 注入精英
            new_pop = np.array(new_pop, dtype=np.float32)
            new_pop[:k_freeze] = elites

            # 评估新种群
            new_cfgs = [_clip_to_space(_vec_to_cfg(new_pop[i], meta), self.bounds) for i in range(self.pop_size)]
            new_fit = np.array([eval_fn(c) for c in new_cfgs], dtype=np.float32)

            # 选择
            pop_vec = new_pop
            pop_cfg = new_cfgs
            fitness = new_fit

            # 更新全局最优
            bidx = int(np.argmin(fitness))
            if fitness[bidx] < best_fit:
                best_fit = float(fitness[bidx])
                best_vec = pop_vec[bidx].copy()
                best_cfg = pop_cfg[bidx]

            if verbose:
                print(f"[RIME] iter {t+1}/{self.iters}: best_val_loss={best_fit:.6f}  best_cfg={best_cfg}")

        return best_cfg, best_fit
