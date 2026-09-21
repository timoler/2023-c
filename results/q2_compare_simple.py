import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

fp = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp, engine='openpyxl', header=1)
expert_cols = df.columns[0:5].tolist()

# Method A: Z-score
def method_zscore(df, cols):
    res = []
    for i, r in df.iterrows():
        z = []
        for c in cols:
            x = pd.to_numeric(r[c], errors='coerce')
            if pd.notna(x):
                col_data = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(col_data) > 10 and col_data.std() > 1e-10:
                    z.append((x - col_data.mean()) / col_data.std())
        if len(z) >= 2:
            res.append({'id': i, 'score_A': np.mean(z), 'n': len(z)})
    return pd.DataFrame(res)

# Method B: Robust Z-score
def method_robust(df, cols):
    res = []
    for i, r in df.iterrows():
        z = []
        for c in cols:
            x = pd.to_numeric(r[c], errors='coerce')
            if pd.notna(x):
                col_data = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(col_data) > 10:
                    med = col_data.median()
                    mad = stats.median_abs_deviation(col_data)
                    if mad > 1e-10:
                        z.append((x - med) / mad)
        if len(z) >= 2:
            res.append({'id': i, 'score_B': np.mean(z), 'n': len(z)})
    return pd.DataFrame(res)

print('Running Method A...')
dfA = method_zscore(df, expert_cols)
print('A done:', len(dfA), 'works')

print('Running Method B...')
dfB = method_robust(df, expert_cols)
print('B done:', len(dfB), 'works')

# Merge and compare
df_cmp = dfA.merge(dfB[['id','score_B']], on='id')
rho, _ = stats.spearmanr(df_cmp['score_A'], df_cmp['score_B'])
print('\nA vs B Spearman:', round(rho, 4))

# Save
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
out = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\question2_multi_method_comparison.csv'
df_cmp.to_csv(out, index=False, encoding='utf-8-sig')
print('Saved:', out)

