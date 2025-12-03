import pandas as pd
from pathlib import Path

# =========================================================
# 路径设置（可直接运行）
# =========================================================
data_folder = Path("./data/raw data")     # 原始数据路径
output_folder = Path("./data/cleaned")    # 输出路径
output_folder.mkdir(parents=True, exist_ok=True)

# 参数设置
FREQ_MIN = 15          # 数据采样频率（分钟）
MAX_GAP_MIN = 120      # 最大连续插值缺口（分钟）
LIMIT_STEPS = MAX_GAP_MIN // FREQ_MIN     # 最大插值步数（以行计）

# =========================================================
# 工具函数
# =========================================================
def _detect_datetime_col(df: pd.DataFrame):
    """自动识别时间列名"""
    candidates = ["datetime", "DateTime", "timestamp", "Timestamp", "time", "Time", df.columns[0]]
    for col in candidates:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().mean() > 0.9:
                return col
    return None


def fill_missing_values_time_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充方法（适合 15min 环境监测数据）
    步骤：
        1. 时间相邻插值（≤2小时）
        2. 相位均值（同一时间槽位）
        3. 前向/后向填充兜底
    """
    df = df.copy()

    # --- 识别时间列并设为索引 ---
    tcol = _detect_datetime_col(df)
    if tcol is not None:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
        has_time_index = True
    else:
        has_time_index = False

    # --- 八个主要变量列 ---
    cols = ["Temp", "SpCond", "Sal", "DO_pct", "DO_mgl", "Depth", "pH", "Turb"]
    cols = [c for c in cols if c in df.columns]  # 若部分列缺失则跳过

    # --- 1) 时间相邻插值 ---
    for col in cols:
        if has_time_index:
            df[col] = df[col].interpolate(
                method="time",
                limit=LIMIT_STEPS,
                limit_direction="both"
            )
        else:
            df[col] = df[col].interpolate(
                method="linear",
                limit=LIMIT_STEPS,
                limit_direction="both"
            )

    # --- 2) 相位均值填补 ---
    if has_time_index:
        slot = df.index.hour * 60 + df.index.minute
        for col in cols:
            phase_mean = df.groupby(slot)[col].transform("mean")
            df[col] = df[col].fillna(phase_mean)

    # --- 3) 兜底前后向填充 ---
    df[cols] = df[cols].ffill().bfill()

    # --- 恢复时间列 ---
    if has_time_index:
        df = df.reset_index()

    return df


# =========================================================
# 主处理流程
# =========================================================
excel_files = sorted(data_folder.glob("*.xlsx"))
if not excel_files:
    print(" 未在 data/raw data 文件夹中找到任何 Excel 文件，请检查路径。")
else:
    print(f"共检测到 {len(excel_files)} 个文件。开始处理...\n")

    for file_path in excel_files:
        if file_path.name.startswith("~$"):
            continue  # 跳过临时文件

        print(f" 正在处理：{file_path.name}")
        df = pd.read_excel(file_path)

        # 缺失统计
        print("缺失统计：")
        print(df.isnull().sum(), "\n")

        # 填充缺失值
        df_cleaned = fill_missing_values_time_phase(df)

        # 保存结果
        save_path = output_folder / f"cleaned_{file_path.name}"
        df_cleaned.to_excel(save_path, index=False)
        print(f" 已保存到：{save_path}\n")

    print(" 所有文件已处理完成！")
