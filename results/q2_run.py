import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

print("=== C题第二问：标准分计算模型分析 ===\n")

fp2_1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp2_1, engine='openpyxl', header=1)

# 专家分数列索引
expert_score_cols = [6, 9, 12, 15, 18, 25, 28, 31]
expert_names = ['专家一', '专家二', '专家三', '专家四', '专家五', '专家一.1', '专家二.1', '专家三.1']

# 提取评分数据
data = {}
for idx, col_idx in enumerate(expert_score_cols):
    name = expert_names[idx]
    scores = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce')
    valid = scores.notna()
    data[name] = scores[valid].values
    if len(scores[valid]) > 0:
        print(f'{name}: n={valid.sum()}, range=[{scores.min():.1f}, {scores.max():.1f}]')

# 专家统计量
expert_stats = {}
for name, scores in data.items():
    if len(scores) > 10:
        expert_stats[name] = {
            'mean': np.mean(scores), 'std': np.std(scores, ddof=1),
            'median': np.median(scores),
            'mad': np.median(np.abs(scores - np.median(scores))) * 1.4826,
            'skew': stats.skew(scores), 'n': len(scores)
        }

# 标准化函数
def compute_scores(method='classic'):
    results = []
    for row_idx in range(2, len(df)):
        z_vals = []
        for name, col_idx in zip(expert_names, expert_score_cols):
            if name in expert_stats:
                val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                if pd.notna(val):
                    if method == 'classic':
                        z = (val - expert_stats[name]['mean']) / expert_stats[name]['std']
                    elif method == 'robust':
                        mad = expert_stats[name]['mad'] if expert_stats[name]['mad'] > 1e-10 else 1.0
                        z = (val - expert_stats[name]['median']) / mad
                    else:
                        all_v = data[name]
                        z = (np.sum(all_v <= val) / len(all_v)) * 100
                    z_vals.append(z)
        results.append(np.mean(z_vals) if len(z_vals) >= 2 else np.nan)
    return np.array(results)

scores_z = compute_scores('classic')
scores_robust = compute_scores('robust')
scores_pct = compute_scores('percentile')

# 新模型: 加权Z-score
weights = {}
for name, s in expert_stats.items():
    if abs(s['mean']) > 1e-10:
        cv = s['std'] / abs(s['mean'])
        weights[name] = 1.0 / (1 + cv)
    else:
        weights[name] = 0.5

scores_cons = []
for row_idx in range(2, len(df)):
    z_sum, w_sum = 0, 0
    for name, col_idx in zip(expert_names, expert_score_cols):
        if name in expert_stats and name in weights:
            val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
            if pd.notna(val) and expert_stats[name]['std'] > 1e-10:
                z = (val - expert_stats[name]['mean']) / expert_stats[name]['std']
                w = weights[name]
                z_sum += w * z
                w_sum += w
    scores_cons.append(z_sum/w_sum if w_sum > 1e-10 else np.nan)
scores_cons = np.array(scores_cons)

# 原始总分
orig = []
for row_idx in range(2, len(df)):
    vals = [pd.to_numeric(df.iloc[row_idx, c], errors='coerce') for c in expert_score_cols]
    vals = [v for v in vals if pd.notna(v)]
    orig.append(np.mean(vals) if len(vals)>=2 else np.nan)
orig = np.array(orig)

# 验证
valid = ~np.isnan(scores_z)
print('\n=== 方案对比 ===')
for name, scores in [('Original',orig),('Z-score',scores_z),('Robust',scores_robust),('Percentile',scores_pct/100),('Consensus(新)',scores_cons)]:
    v = scores[valid]
    if len(v) > 10:
        ks = stats.kstest(v, 'norm')[1]
        rho, _ = stats.spearmanr(orig[valid], v)
        print(f'{name:15s} | mean={np.mean(v):+.3f} std={np.std(v):.3f} KS-p={ks:.3f} rho={rho:.3f}')

# 排名重合
print('\n=== Top-20重合度 ===')
t20_o = set(np.argsort(-orig[valid])[:20])
for name, sc in [('Z-score',scores_z),('Robust',scores_robust),('Consensus',scores_cons)]:
    t20 = set(np.argsort(-sc[valid])[:20])
    print(f'{name:15s}: {len(t20_o & t20)}/20')

# 输出
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
out = pd.DataFrame({'idx':range(len(df)-2),'orig':orig,'z':scores_z,'robust':scores_robust,'pct':scores_pct,'cons':scores_cons})
out.to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_results.csv', index=False, encoding='utf-8-sig')

with open(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_formula.txt', 'w', encoding='utf-8') as f:
    f.write("新模型: S_i = sum_j[w_j*(x_ij-mu_j)/sigma_j] / sum_j[w_j]\n")
    f.write("权重: w_j = 1/(1+CV_j), CV_j=sigma_j/|mu_j|\n")
    f.write("专家权重: " + str({k:round(v,3) for k,v in weights.items()}) + "\n")

print('\n输出: results/q2_results.csv, q2_formula.txt')
print('完成!')
