import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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

z_a = zscore_std(df, expert_cols)
z_b = robust_zscore_std(df, expert_cols) 
z_c = percentile_std(df, expert_cols)
orig = df[expert_cols].apply(lambda r: pd.to_numeric(r, errors='coerce').dropna().mean(), axis=1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('C题第二问：标准分计算方法对比', fontsize=14, fontweight='bold')

# 子图1
ax = axes[0,0]
ax.scatter(orig.rank(), z_a.rank(), alpha=0.3, s=10, c='steelblue')
rho_a, _ = stats.spearmanr(orig, z_a)
ax.set_xlabel('原始总分排名')
ax.set_ylabel('Z-score标准分排名')
ax.set_title('方法A: Z-score (rho=' + str(round(rho_a,3)) + ')')
ax.grid(alpha=0.3)

# 子图2
ax = axes[0,1]
ax.scatter(z_a, z_b, alpha=0.3, s=10, c='coral')
rho_ab, _ = stats.spearmanr(z_a, z_b)
ax.set_xlabel('Z-score标准分')
ax.set_ylabel('Robust Z-score标准分')
ax.set_title('方法A vs B (rho=' + str(round(rho_ab,3)) + ')')
ax.grid(alpha=0.3)

# 子图3
ax = axes[1,0]
data_box = [z_a.dropna(), z_b.dropna(), z_c.dropna()/100]
ax.boxplot(data_box, labels=['Z-score', 'Robust', 'Percentile/100'], patch_artist=True)
ax.set_ylabel('标准分值')
ax.set_title('标准分分布对比')
ax.grid(alpha=0.3, axis='y')

# 子图4
top50_idx = orig.dropna().nlargest(50).index
heat_data = pd.DataFrame({
    'Original': orig.loc[top50_idx].rank(ascending=False),
    'Z-score': z_a.loc[top50_idx].rank(ascending=False), 
    'Robust': z_b.loc[top50_idx].rank(ascending=False),
    'Percentile': z_c.loc[top50_idx].rank(ascending=False)
})
im = axes[1,1].imshow(heat_data.values.T, cmap='RdYlGn_r', aspect='auto')
axes[1,1].set_yticks(range(4))
axes[1,1].set_yticklabels(['Original', 'Z-score', 'Robust', 'Percentile'])
axes[1,1].set_xticks(range(50))
axes[1,1].set_xticklabels(range(1,51), rotation=90, fontsize=6)
axes[1,1].set_title('Top 50作品排名对比')
axes[1,1].grid(False)
plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

plt.tight_layout()
out_fig = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_method_comparison.png'
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print('Figure saved:', out_fig)

# 输出指标
print('\n方法对比关键指标:')
ks_a = stats.kstest(z_a.dropna(), 'norm')[1]
ks_b = stats.kstest(z_b.dropna(), 'norm')[1]
ks_c = stats.kstest(z_c.dropna()/100, 'norm')[1]
print('Z-score:     mean=%.4f, std=%.4f, KS-p=%.3f' % (z_a.mean(), z_a.std(), ks_a))
print('Robust:      mean=%.4f, std=%.4f, KS-p=%.3f' % (z_b.mean(), z_b.std(), ks_b))
print('Percentile:  mean=%.2f, std=%.2f, KS-p=%.3f' % (z_c.mean(), z_c.std(), ks_c))
