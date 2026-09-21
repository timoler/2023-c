import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

# Load data
fp = r"E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx"
df = pd.read_excel(fp, engine="openpyxl", header=1)

# 获取专家评分列（前5列是专家评分）
expert_cols = df.columns[0:5].tolist()
print("专家评分列:", expert_cols)

# Step 1: Calculate expert-level statistics for standardization
expert_stats = {}
for col in expert_cols:
    scores = pd.to_numeric(df[col], errors="coerce")
    valid = scores.dropna()
    if len(valid) > 10:
        expert_stats[col] = {"mu": valid.mean(), "sigma": valid.std()}
        print(f"{col}: mu={expert_stats[col]['mu']:.2f}, sigma={expert_stats[col]['sigma']:.2f}, n={len(valid)}")

# Step 2: Standardize each expert's scores to Z-scores, then aggregate per work
results = []
for idx, row in df.iterrows():
    z_scores = []
    for col in expert_cols:
        raw = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(raw) and col in expert_stats:
            mu, sigma = expert_stats[col]["mu"], expert_stats[col]["sigma"]
            if sigma > 1e-10:
                z_scores.append((raw - mu) / sigma)
    if len(z_scores) >= 2:
        results.append({"work_id": idx, "standard_score": np.mean(z_scores), "n_experts": len(z_scores)})

df_std = pd.DataFrame(results)
print(f"\nValid works: {len(df_std)}")
print(f"Standard score range: [{df_std['standard_score'].min():.3f}, {df_std['standard_score'].max():.3f}]")
print(f"Mean: {df_std['standard_score'].mean():.4f}, Std: {df_std['standard_score'].std():.4f}")

print("\nTop 10 by standard score:")
print(df_std.nlargest(10, "standard_score")[["work_id","standard_score","n_experts"]].to_string(index=False))

# Save results
os.makedirs(r"E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results", exist_ok=True)
out_csv = r"E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\question2_standard_scores.csv"
df_std.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_csv}")
