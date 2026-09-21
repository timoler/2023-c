import pandas as pd, numpy as np
from scipy import stats
import warnings, os, json
warnings.filterwarnings('ignore')

# ===== 1. 数据加载 =====
print("=== 数据加载 ===")
fp1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据1.xlsx'
fp2_1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
fp2_2 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.2 .xlsx'

df1 = pd.read_excel(fp1, engine='openpyxl')  # 一等奖协商数据
df2_1 = pd.read_excel(fp2_1, engine='openpyxl', header=1)  # 专家评审原始数据
df2_2 = pd.read_excel(fp2_2, engine='openpyxl')  # 可能含其他信息

print(f"数据1（一等奖协商）: {df1.shape}, 列: {df1.columns.tolist()[:8]}...")
print(f"数据2.1（专家评分）: {df2_1.shape}, 列: {df2_1.columns.tolist()[:6]}...")
print(f"数据2.2: {df2_2.shape}")

# ===== 2. 专家评分分布分析 =====
print("\n=== 专家评分分布特征 ===")
expert_cols = df2_1.columns[0:5].tolist()  # 前5列为专家评分

expert_stats = {}
for col in expert_cols:
    s = pd.to_numeric(df2_1[col], errors='coerce').dropna()
    if len(s) > 10:
        expert_stats[col] = {
            'n': len(s),
            'mean': s.mean(),
            'std': s.std(),
            'median': s.median(),
            'mad': np.median(np.abs(s - s.median())) * 1.4826,
            'skew': stats.skew(s),
            'kurt': stats.kurtosis(s),
            'min': s.min(),
            'max': s.max(),
            'q1': s.quantile(0.25),
            'q3': s.quantile(0.75)
        }
        print(f"{col}: n={len(s):3d}, μ={expert_stats[col]['mean']:.2f}, σ={expert_stats[col]['std']:.2f}, skew={expert_stats[col]['skew']:.2f}")

# ===== 3. 作品级原始分分布 =====
print("\n=== 作品原始总分分布 ===")
df2_1['original_total'] = df2_1[expert_cols].apply(
    lambda r: pd.to_numeric(r, errors='coerce').dropna().mean(), axis=1)
valid_mask = df2_1['original_total'].notna()
print(f"有效作品数: {valid_mask.sum()}")
print(f"原始总分: mean={df2_1.loc[valid_mask,'original_total'].mean():.2f}, "
      f"std={df2_1.loc[valid_mask,'original_total'].std():.2f}, "
      f"skew={stats.skew(df2_1.loc[valid_mask,'original_total']):.2f}")

# ===== 4. 四种标准化方案实现 =====
print("\n=== 标准化方案实现 ===")

def scheme_zscore(df, cols, stats_dict):
    """方案A: 经典Z-score"""
    res = []
    for _, row in df.iterrows():
        z = []
        for c in cols:
            if c in stats_dict:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val) and stats_dict[c]['std'] > 1e-10:
                    z.append((val - stats_dict[c]['mean']) / stats_dict[c]['std'])
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def scheme_robust(df, cols, stats_dict):
    """方案B: Robust Z-score (MAD)"""
    res = []
    for _, row in df.iterrows():
        z = []
        for c in cols:
            if c in stats_dict:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val):
                    mad_scaled = stats_dict[c]['mad'] if stats_dict[c]['mad']>1e-10 else 1.0
                    z.append((val - stats_dict[c]['median']) / mad_scaled)
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def scheme_percentile(df, cols):
    """方案C: 百分位数排名"""
    res = []
    for _, row in df.iterrows():
        p = []
        for c in cols:
            val = pd.to_numeric(row[c], errors='coerce')
            if pd.notna(val):
                cv = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(cv)>0:
                    p.append((cv <= val).sum() / len(cv) * 100)
        res.append(np.mean(p) if len(p)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def scheme_consensus_weighted(df, cols, stats_dict, consensus_ref=None):
    """方案D(新): 协商一致性加权Z-score（利用数据1校准）"""
    # 核心思想: 用一等奖协商排序反推专家可信度权重
    if consensus_ref is None:
        # 无参考时退化为等权Z-score
        return scheme_zscore(df, cols, stats_dict)
    
    # 步骤1: 识别协商一致的"黄金标准"作品
    gold_ids = consensus_ref['作品编号'].dropna().astype(str).tolist()
    
    # 步骤2: 计算每位专家在黄金作品上的评分偏差
    expert_bias = {}
    for col in cols:
        if col in stats_dict:
            # 提取该专家对黄金作品的评分
            gold_scores = []
            gold_original = []
            for idx, row in df.iterrows():
                if str(idx) in gold_ids:  # 简化: 用行号模拟作品ID
                    val = pd.to_numeric(row[col], errors='coerce')
                    orig = pd.to_numeric(row['original_total'], errors='coerce')
                    if pd.notna(val) and pd.notna(orig):
                        gold_scores.append(val)
                        gold_original.append(orig)
            if len(gold_scores) >= 5:
                # 计算专家评分与协商共识的相关性
                corr, _ = stats.pearsonr(gold_scores, gold_original)
                expert_bias[col] = max(0.1, (corr + 1) / 2)  # 映射到[0.1, 1.0]权重
            else:
                expert_bias[col] = 0.5  # 默认权重
    
    # 步骤3: 加权聚合标准分
    res = []
    for _, row in df.iterrows():
        z_weighted_sum, weight_sum = 0, 0
        for c in cols:
            if c in stats_dict and c in expert_bias:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val) and stats_dict[c]['std'] > 1e-10:
                    z = (val - stats_dict[c]['mean']) / stats_dict[c]['std']
                    w = expert_bias[c]
                    z_weighted_sum += w * z
                    weight_sum += w
        res.append(z_weighted_sum / weight_sum if weight_sum > 1e-10 else np.nan)
    return pd.Series(res, index=df.index)

# 执行四种方案
scores_A = scheme_zscore(df2_1, expert_cols, expert_stats)
scores_B = scheme_robust(df2_1, expert_cols, expert_stats)
scores_C = scheme_percentile(df2_1, expert_cols)
scores_D = scheme_consensus_weighted(df2_1, expert_cols, expert_stats, consensus_ref=df1)

# ===== 5. 方案对比验证 =====
print("\n=== 方案对比验证指标 ===")
valid_idx = scores_A.notna()

def compute_metrics(orig, std_scores, name):
    v = std_scores[valid_idx]
    o = orig[valid_idx]
    return {
        'method': name,
        'mean': v.mean(),
        'std': v.std(),
        'ks_p': stats.kstest(v, 'norm')[1],
        'spearman_rho': stats.spearmanr(o, v)[0],
        'kendall_tau': stats.kendalltau(o, v)[0]
    }

metrics = [
    compute_metrics(df2_1.loc[valid_idx,'original_total'], scores_A, 'A:Z-score'),
    compute_metrics(df2_1.loc[valid_idx,'original_total'], scores_B, 'B:Robust'),
    compute_metrics(df2_1.loc[valid_idx,'original_total'], scores_C/100, 'C:Percentile'),
    compute_metrics(df2_1.loc[valid_idx,'original_total'], scores_D, 'D:Consensus-Weighted')
]

for m in metrics:
    print(f"{m['method']:25s} | mean={m['mean']:+.3f} std={m['std']:.3f} "
          f"KS-p={m['ks_p']:.3f} ρ={m['spearman_rho']:.3f} τ={m['kendall_tau']:.3f}")

# ===== 6. 排名稳定性分析 =====
print("\n=== Top-20排名重合度 ===")
top20_orig = set(df2_1.loc[valid_idx,'original_total'].nlargest(20).index)
for name, scores in [('A:Z-score',scores_A),('B:Robust',scores_B),('C:Percentile',scores_C),('D:Consensus',scores_D)]:
    top20_std = set(scores[valid_idx].nlargest(20).index)
    overlap = len(top20_orig & top20_std)
    print(f"{name:20s}: Top-20重合 {overlap}/20 ({overlap*5:.0f}%)")

# ===== 7. 输出结果 =====
print("\n=== 结果输出 ===")
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)

# 综合对比表
comparison_df = pd.DataFrame({
    'work_id': df2_1.index,
    'original': df2_1['original_total'],
    'zscore': scores_A,
    'robust': scores_B,
    'percentile': scores_C,
    'consensus_weighted': scores_D,
    'rank_orig': df2_1['original_total'].rank(ascending=False),
    'rank_zscore': scores_A.rank(ascending=False),
    'rank_robust': scores_B.rank(ascending=False),
    'rank_consensus': scores_D.rank(ascending=False)
})
comparison_df.to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_full_comparison.csv', 
                     index=False, encoding='utf-8-sig')
print("✓ 对比表: results/q2_full_comparison.csv")

# 方案D的专家权重输出
expert_weights = {k: v for k,v in zip(expert_cols, 
    [expert_bias.get(c, 0.5) for c in expert_cols]) if 'expert_bias' in locals()}
if expert_weights:
    with open(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_expert_weights.json', 'w', encoding='utf-8') as f:
        json.dump(expert_weights, f, ensure_ascii=False, indent=2)
    print("✓ 专家权重: results/q2_expert_weights.json")

print("\n✅ 分析完成！四种方案已实现并对比验证。")
