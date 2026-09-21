import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

fp = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp, engine='openpyxl', header=1)
expert_cols = df.columns[0:5].tolist()

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

def method_percentile(df, cols):
    res = []
    for i, r in df.iterrows():
        p = []
        for c in cols:
            x = pd.to_numeric(r[c], errors='coerce')
            if pd.notna(x):
                col_data = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(col_data) > 10:
                    # Compute percentile rank using pandas
                    all_vals = pd.concat([col_data, pd.Series([x])], ignore_index=True)
                    rank = stats.rankdata(all_vals, method='average')[-1]
                    p.append(rank / (len(col_data) + 1) * 100)
        if len(p) >= 2:
            res.append({'id': i, 'score_C': np.mean(p), 'n': len(p)})
    return pd.DataFrame(res)

print('Method A (Z-score)...')
dfA = method_zscore(df, expert_cols)
print('  Count:', len(dfA))

print('Method B (Robust)...')
dfB = method_robust(df, expert_cols)
print('  Count:', len(dfB))

print('Method C (Percentile)...')
dfC = method_percentile(df, expert_cols)
print('  Count:', len(dfC))

df_cmp = dfA.merge(dfB[['id','score_B']], on='id').merge(dfC[['id','score_C']], on='id')

print('\n=== Metrics ===')
for col, name in [('score_A','A'), ('score_B','B'), ('score_C','C')]:
    m, s = df_cmp[col].mean(), df_cmp[col].std()
    print(name + ': mean=' + str(round(m,3)) + ', std=' + str(round(s,3)))

rho_AB, _ = stats.spearmanr(df_cmp['score_A'], df_cmp['score_B'])
rho_AC, _ = stats.spearmanr(df_cmp['score_A'], df_cmp['score_C'])
rho_BC, _ = stats.spearmanr(df_cmp['score_B'], df_cmp['score_C'])
print('\nSpearman consistency:')
print('A-B:', round(rho_AB,4), 'A-C:', round(rho_AC,4), 'B-C:', round(rho_BC,4))

top10_A = set(df_cmp.nlargest(10, 'score_A')['id'])
top10_B = set(df_cmp.nlargest(10, 'score_B')['id'])
top10_C = set(df_cmp.nlargest(10, 'score_C')['id'])
print('\nTop-10 overlap: A-B=' + str(len(top10_A & top10_B)) + '/10, A-C=' + str(len(top10_A & top10_C)) + '/10')

os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
out = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\question2_multi_method_comparison.csv'
df_cmp.to_csv(out, index=False, encoding='utf-8-sig')
print('\nSaved:', out)

