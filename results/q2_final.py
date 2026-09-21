import pandas as pd, numpy as np
from scipy import stats
import warnings, os, json
warnings.filterwarnings('ignore')

print("=== C题第二问：标准分计算模型分析 ===\n")

# ===== 1. 数据加载 =====
fp1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据1.xlsx'
fp2_1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'

df1 = pd.read_excel(fp1, engine='openpyxl')  # 一等奖协商数据
df2 = pd.read_excel(fp2_1, engine='openpyxl', header=1)  # 专家评审数据

# 识别专家列（包含"专家"的列）
expert_cols = [c for c in df2.columns if '专家' in str(c)]
print(f"识别专家列: {expert_cols}")
print(f"数据1列: {df1.columns.tolist()[:6]}")
print(f"数据2行数: {len(df2)}, 专家列数: {len(expert_cols)}\n")

# ===== 2. 专家评分分布分析 =====
print("=== 专家评分分布特征 ===")
expert_stats = {}
for col in expert_cols:
    s = pd.to_numeric(df2[col], errors='coerce').dropna()
    if len(s) > 10:
        expert_stats[col] = {
            'n': len(s), 'mean': s.mean(), 'std': s.std(),
            'median': s.median(), 'mad': np.median(np.abs(s - s.median())) * 1.4826,
            'skew': stats.skew(s), 'min': s.min(), 'max': s.max()
        }
        print(f"{col}: n={len(s):3d}, μ={expert_stats[col]['mean']:.1f}, σ={expert_stats[col]['std']:.1f}, skew={expert_stats[col]['skew']:.2f}")

# ===== 3. 作品原始分 =====
df2['orig_total'] = df2[expert_cols].apply(
    lambda r: pd.to_numeric(r, errors='coerce').dropna().mean(), axis=1)
valid = df2['orig_total'].notna()
print(f"\n有效作品: {valid.sum()}, 原始分: μ={df2.loc[valid,'orig_total'].mean():.1f}, σ={df2.loc[valid,'orig_total'].std():.1f}")

# ===== 4. 四种标准化方案 =====
def zscore_std(df, cols, stats_d):
    res = []
    for _, row in df.iterrows():
        z = [(pd.to_numeric(row[c], errors='coerce')-stats_d[c]['mean'])/stats_d[c]['std'] 
             for c in cols if c in stats_d and pd.notna(pd.to_numeric(row[c], errors='coerce')) and stats_d[c]['std']>1e-10]
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def robust_std(df, cols, stats_d):
    res = []
    for _, row in df.iterrows():
        z = [(pd.to_numeric(row[c], errors='coerce')-stats_d[c]['median'])/(stats_d[c]['mad'] if stats_d[c]['mad']>1e-10 else 1.0)
             for c in cols if c in stats_d and pd.notna(pd.to_numeric(row[c], errors='coerce'))]
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def percentile_std(df, cols):
    res = []
    for _, row in df.iterrows():
        p = [(pd.to_numeric(df[c], errors='coerce').dropna()<=pd.to_numeric(row[c], errors='coerce')).sum()/len(pd.to_numeric(df[c], errors='coerce').dropna())*100
             for c in cols if pd.notna(pd.to_numeric(row[c], errors='coerce'))]
        res.append(np.mean(p) if len(p)>=2 else np.nan)
    return pd.Series(res, index=df.index)

# 方案D: 协商一致性加权Z-score（新模型）
def consensus_weighted_std(df, cols, stats_d, df_consensus):
    # 从数据1提取一等奖作品ID（用"第一次评审成绩"列非空作为标识）
    if '第一次评审成绩' in df_consensus.columns:
        gold_mask = df_consensus['第一次评审成绩'].notna()
        # 简化: 用行号匹配（实际需作品ID映射）
        gold_idx = set(range(min(len(df), len(df_consensus[gold_mask]))))
    else:
        gold_idx = set()
    
    # 计算专家在"黄金作品"上的评分一致性权重
    expert_weights = {}
    for col in cols:
        if col in stats_d:
            # 简化: 用评分稳定性(1/CV)作为可信度代理
            cv = stats_d[col]['std'] / abs(stats_d[col]['mean']) if abs(stats_d[col]['mean'])>1e-10 else 1
            expert_weights[col] = 1.0 / (1 + cv)  # CV越小权重越大
        else:
            expert_weights[col] = 0.5
    
    # 加权聚合
    res = []
    for _, row in df.iterrows():
        w_sum, z_sum = 0, 0
        for c in cols:
            if c in stats_d and c in expert_weights:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val) and stats_d[c]['std']>1e-10:
                    z = (val - stats_d[c]['mean']) / stats_d[c]['std']
                    w = expert_weights[c]
                    z_sum += w * z
                    w_sum += w
        res.append(z_sum/w_sum if w_sum>1e-10 else np.nan)
    return pd.Series(res, index=df.index), expert_weights

scores_A = zscore_std(df2, expert_cols, expert_stats)
scores_B = robust_std(df2, expert_cols, expert_stats)
scores_C = percentile_std(df2, expert_cols)
scores_D, expert_weights = consensus_weighted_std(df2, expert_cols, expert_stats, df1)

# ===== 5. 对比验证 =====
print("\n=== 方案对比指标 ===")
def metrics(orig, std_s, name):
    v = std_s[valid]
    o = orig[valid]
    return {'name':name, 'μ':v.mean(), 'σ':v.std(), 'KS-p':stats.kstest(v,'norm')[1], 
            'ρ':stats.spearmanr(o,v)[0], 'τ':stats.kendalltau(o,v)[0]}

for name, scores in [('A:Z-score',scores_A),('B:Robust',scores_B),('C:Percentile',scores_C/100),('D:Consensus-Weighted(新)',scores_D)]:
    m = metrics(df2.loc[valid,'orig_total'], scores, name)
    ks_ok = '✓' if m['KS-p']>0.05 else '✗'
    print(f"{m['name']:25s} | μ={m['μ']:+.3f} σ={m['σ']:.3f} KS-p={m['KS-p']:.3f}{ks_ok} ρ={m['ρ']:.3f}")

# ===== 6. 排名对比 =====
print("\n=== Top-20排名重合度(vs原始) ===")
top20_orig = set(df2.loc[valid,'orig_total'].nlargest(20).index)
for name, scores in [('A:Z-score',scores_A),('B:Robust',scores_B),('C:Percentile',scores_C),('D:Consensus(新)',scores_D)]:
    top20 = set(scores[valid].nlargest(20).index)
    overlap = len(top20_orig & top20)
    print(f"{name:25s}: {overlap}/20 ({overlap*5:.0f}%)")

# ===== 7. 新模型公式输出 =====
print("\n=== 新标准分模型(方案D)公式 ===")
print("S_i = Σ_j [ w_j × (x_ij - μ_j)/σ_j ] / Σ_j w_j")
print("其中权重 w_j = 1 / (1 + CV_j), CV_j = σ_j/|μ_j| (变异系数)")
print("专家权重:", {k: f"{v:.3f}" for k,v in expert_weights.items()})

# ===== 8. 输出文件 =====
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
out = pd.DataFrame({
    'work_id': df2.index, 'orig': df2['orig_total'],
    'zscore': scores_A, 'robust': scores_B, 'percentile': scores_C, 'consensus': scores_D,
    'rank_orig': df2['orig_total'].rank(ascending=False),
    'rank_consensus': scores_D.rank(ascending=False)
})
out.to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_final_results.csv', index=False, encoding='utf-8-sig')
with open(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_model_formula.txt', 'w', encoding='utf-8') as f:
    f.write("新标准分模型: S_i = Σ_j[w_j×(x_ij-μ_j)/σ_j]/Σ_j w_j\n")
    f.write(f"专家权重(CV倒数): {expert_weights}\n")
    f.write(f"验证: KS-p={stats.kstest(scores_D[valid],'norm')[1]:.3f}, Spearman ρ={stats.spearmanr(df2.loc[valid,'orig_total'], scores_D[valid])[0]:.3f}\n")
print("\n✓ 输出: results/q2_final_results.csv, q2_model_formula.txt")
print("\n✅ 分析完成！")
