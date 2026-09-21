import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

# 加载数据
fp = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp, engine='openpyxl', header=1)
expert_cols = df.columns[0:5].tolist()

def zscore_std(df, cols):
    stats_d = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(s)>10:
            stats_d[c] = (s.mean(), s.std())
    res = []
    for _, row in df.iterrows():
        z = []
        for c in cols:
            if c in stats_d:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val) and stats_d[c][1]>1e-10:
                    z.append((val - stats_d[c][0])/stats_d[c][1])
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def robust_zscore_std(df, cols):
    stats_d = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(s)>10:
            med = s.median()
            mad = np.median(np.abs(s-med))*1.4826
            stats_d[c] = (med, mad if mad>1e-10 else 1.0)
    res = []
    for _, row in df.iterrows():
        z = []
        for c in cols:
            if c in stats_d:
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val):
                    z.append((val - stats_d[c][0])/stats_d[c][1])
        res.append(np.mean(z) if len(z)>=2 else np.nan)
    return pd.Series(res, index=df.index)

def percentile_std(df, cols):
    res = []
    for _, row in df.iterrows():
        p = []
        for c in cols:
            val = pd.to_numeric(row[c], errors='coerce')
            if pd.notna(val):
                cv = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(cv)>0:
                    p.append((cv<=val).sum()/len(cv)*100)
        res.append(np.mean(p) if len(p)>=2 else np.nan)
    return pd.Series(res, index=df.index)

# 计算
z_a = zscore_std(df, expert_cols)
z_b = robust_zscore_std(df, expert_cols) 
z_c = percentile_std(df, expert_cols)
orig = df[expert_cols].apply(lambda r: pd.to_numeric(r, errors='coerce').dropna().mean(), axis=1)

# 构建对比表
comparison = pd.DataFrame({
    'work_id': range(len(df)),
    'original_score': orig,
    'zscore': z_a,
    'robust_zscore': z_b,
    'percentile': z_c,
    'rank_original': orig.rank(ascending=False),
    'rank_zscore': z_a.rank(ascending=False),
    'rank_robust': z_b.rank(ascending=False),
    'rank_percentile': z_c.rank(ascending=False),
})

# 计算验证指标
valid_mask = comparison['zscore'].notna()
ks_a = stats.kstest(comparison.loc[valid_mask, 'zscore'], 'norm')[1]
ks_b = stats.kstest(comparison.loc[valid_mask, 'robust_zscore'], 'norm')[1]
ks_c = stats.kstest(comparison.loc[valid_mask, 'percentile']/100, 'norm')[1]
rho_ab = stats.spearmanr(comparison['zscore'], comparison['robust_zscore'])[0]
rho_ac = stats.spearmanr(comparison['zscore'], comparison['percentile'])[0]
rho_bc = stats.spearmanr(comparison['robust_zscore'], comparison['percentile'])[0]

# 添加指标汇总行
summary = pd.DataFrame({
    'work_id': ['SUMMARY'],
    'original_score': [comparison['original_score'].mean()],
    'zscore': [comparison['zscore'].mean()],
    'robust_zscore': [comparison['robust_zscore'].mean()],
    'percentile': [comparison['percentile'].mean()],
    'rank_original': [np.nan],
    'rank_zscore': [np.nan],
    'rank_robust': [np.nan],
    'rank_percentile': [np.nan],
})
comparison = pd.concat([comparison, summary], ignore_index=True)

# 保存
out_csv = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_method_comparison.csv'
comparison.to_csv(out_csv, index=False, encoding='utf-8-sig')
print('对比表已保存:', out_csv)

# 输出关键指标
print('\n=== 方法对比验证指标 ===')
print('分布正态性(KS检验p值):')
print('  Z-score:     %.3f %s' % (ks_a, '✓' if ks_a>0.05 else '✗'))
print('  Robust:      %.3f %s' % (ks_b, '✓' if ks_b>0.05 else '✗'))
print('  Percentile:  %.3f %s' % (ks_c, '✓' if ks_c>0.05 else '✗'))
print('\n方法间Spearman相关性:')
print('  Z-score vs Robust:    %.3f' % rho_ab)
print('  Z-score vs Percentile: %.3f' % rho_ac)
print('  Robust vs Percentile:  %.3f' % rho_bc)
print('\n有效作品数:', valid_mask.sum())
