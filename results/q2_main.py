import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

print("=== C题第二问分析启动 ===")

fp = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp, engine='openpyxl', header=1)

expert_cols = [6, 9, 12, 15, 18, 25, 28, 31]
expert_names = ['E1','E2','E3','E4','E5','E1b','E2b','E3b']

all_scores_by_expert = {}
for idx, col in enumerate(expert_cols):
    name = expert_names[idx]
    scores = pd.to_numeric(df.iloc[2:, col], errors='coerce').dropna().values
    if len(scores) > 10:
        all_scores_by_expert[name] = scores
        print(f"{name}: n={len(scores)}, mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

expert_stats = {}
for name, scores in all_scores_by_expert.items():
    expert_stats[name] = {
        'mu': np.mean(scores),
        'sigma': np.std(scores, ddof=1),
        'median': np.median(scores),
        'mad': np.median(np.abs(scores - np.median(scores))) * 1.4826,
        'skew': stats.skew(scores)
    }

orig_scores = []
for row_idx in range(2, len(df)):
    vals = [pd.to_numeric(df.iloc[row_idx, c], errors='coerce') for c in expert_cols]
    vals = [v for v in vals if pd.notna(v)]
    orig_scores.append(np.mean(vals) if len(vals)>=2 else np.nan)
orig_scores = np.array(orig_scores)

def calc_standardized(method='zscore'):
    results = []
    for row_idx in range(2, len(df)):
        z_vals = []
        for name, col in zip(expert_names, expert_cols):
            if name not in expert_stats: continue
            val = pd.to_numeric(df.iloc[row_idx, col], errors='coerce')
            if pd.notna(val):
                s = expert_stats[name]
                if method == 'zscore':
                    z = (val - s['mu']) / s['sigma'] if s['sigma']>1e-10 else 0
                elif method == 'robust':
                    mad = s['mad'] if s['mad']>1e-10 else 1.0
                    z = (val - s['median']) / mad
                elif method == 'percentile':
                    all_v = all_scores_by_expert[name]
                    z = (np.sum(all_v <= val) / len(all_v)) * 100
                z_vals.append(z)
        results.append(np.mean(z_vals) if len(z_vals)>=2 else np.nan)
    return np.array(results)

scores_z = calc_standardized('zscore')
scores_robust = calc_standardized('robust')
scores_pct = calc_standardized('percentile')

weights = {}
for name, s in expert_stats.items():
    cv = s['sigma'] / abs(s['mu']) if abs(s['mu'])>1e-10 else 1
    weights[name] = 1.0 / (1 + cv)

scores_consensus = []
for row_idx in range(2, len(df)):
    z_sum, w_sum = 0, 0
    for name, col in zip(expert_names, expert_cols):
        if name not in expert_stats or name not in weights: continue
        val = pd.to_numeric(df.iloc[row_idx, col], errors='coerce')
        if pd.notna(val) and expert_stats[name]['sigma']>1e-10:
            z = (val - expert_stats[name]['mu']) / expert_stats[name]['sigma']
            w = weights[name]
            z_sum += w * z
            w_sum += w
    scores_consensus.append(z_sum/w_sum if w_sum>1e-10 else np.nan)
scores_consensus = np.array(scores_consensus)

valid = ~np.isnan(scores_z)
print("\n=== 方案对比 (有效样本=%d) ===" % valid.sum())
for name, scores in [('Original',orig_scores),('Z-score',scores_z),('Robust',scores_robust),('Percentile/100',scores_pct/100),('Consensus(新)',scores_consensus)]:
    v = scores[valid]
    if len(v) > 10:
        ks_p = stats.kstest(v, 'norm')[1]
        rho, _ = stats.spearmanr(orig_scores[valid], v)
        ks_mark = 'OK' if ks_p>0.05 else 'NG'
        print(f"{name:20s} | mean={np.mean(v):+.3f} std={np.std(v):.3f} KS-p={ks_p:.3f}[{ks_mark}] rho={rho:.3f}")

print("\n=== Top-20排名重合度(vs原始) ===")
top20_o = set(np.argsort(-orig_scores[valid])[:20])
for name, sc in [('Z-score',scores_z),('Robust',scores_robust),('Consensus(新)',scores_consensus)]:
    top20 = set(np.argsort(-sc[valid])[:20])
    overlap = len(top20_o & top20)
    print(f"{name:20s}: {overlap}/20 ({overlap*5:.0f}%)")

os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
out_df = pd.DataFrame({
    'idx': range(len(df)-2),
    'original': orig_scores,
    'zscore': scores_z,
    'robust': scores_robust,
    'percentile': scores_pct,
    'consensus': scores_consensus
})
out_csv = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_results_final.csv'
out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

formula_txt = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_new_model.txt'
with open(formula_txt, 'w', encoding='utf-8') as f:
    f.write("新标准分模型(协商一致性加权Z-score):\n\n")
    f.write("S_i = sum_j[ w_j * (x_ij - mu_j)/sigma_j ] / sum_j[w_j]\n\n")
    f.write("权重设计: w_j = 1 / (1 + CV_j)\n")
    f.write("  其中 CV_j = sigma_j / |mu_j| (变异系数)\n")
    f.write("  含义: 评分越稳定的专家( CV小 )权重越高\n\n")
    f.write("专家权重值:\n")
    for name, w in weights.items():
        f.write(f"  {name}: {w:.4f}\n")
    v = scores_consensus[valid]
    ks = stats.kstest(v, 'norm')[1]
    rho, _ = stats.spearmanr(orig_scores[valid], v)
    f.write(f"\n验证指标:\n")
    f.write(f"  KS正态性检验: p={ks:.4f} {'[通过]' if ks>0.05 else '[未通过]'}\n")
    f.write(f"  Spearman排名相关: rho={rho:.4f}\n")

print(f"\n输出文件:\n  {out_csv}\n  {formula_txt}")
print("\n=== 分析完成 ===")
