import pandas as pd, numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

print("=== C题第二问：标准分计算模型分析 ===\n")

# ===== 1. 数据加载 =====
fp2_1 = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\数据2.1 .xlsx'
df = pd.read_excel(fp2_1, engine='openpyxl', header=1)

# 专家列结构: 专家N列是作品ID, 下一列是分数
# 列5=专家一(ID), 列6=专家一(分); 列8=专家二(ID), 列9=专家二(分); 等等
expert_score_cols = [6, 9, 12, 15, 18, 25, 28, 31]  # 分数列索引
expert_names = ['专家一', '专家二', '专家三', '专家四', '专家五', '专家一.1', '专家二.1', '专家三.1']

# 提取有效评分数据
data = {}
for idx, col_idx in enumerate(expert_score_cols):
    name = expert_names[idx]
    scores = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce')  # 从第2行开始是数据
    valid = scores.notna()
    data[name] = scores[valid].values
    print(f'{name}: 有效评分 {valid.sum()}个, 范围 [{scores.min():.1f}, {scores.max():.1f}]')

# 构建作品-专家评分矩阵（简化：按行聚合）
print(f'\n总行数: {len(df)-2}')

# 计算每位专家的统计量
expert_stats = {}
for name, scores in data.items():
    if len(scores) > 10:
        expert_stats[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'median': np.median(scores),
            'mad': np.median(np.abs(scores - np.median(scores))) * 1.4826,
            'skew': stats.skew(scores),
            'n': len(scores)
        }
        print(f"{name}: μ={expert_stats[name]['mean']:.1f}, σ={expert_stats[name]['std']:.1f}, skew={expert_stats[name]['skew']:.2f}")

# ===== 2. 四种标准化方案 =====
def compute_zscore(data_dict, stats_d, method='classic'):
    """计算作品的标准分"""
    results = []
    # 按原始DataFrame行迭代，模拟每行有多个专家评分
    for row_idx in range(2, len(df)):
        z_vals = []
        for name, col_idx in zip(expert_names, expert_score_cols):
            if name in stats_d:
                val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                if pd.notna(val):
                    if method == 'classic':
                        z = (val - stats_d[name]['mean']) / stats_d[name]['std']
                    elif method == 'robust':
                        mad = stats_d[name]['mad'] if stats_d[name]['mad'] > 1e-10 else 1.0
                        z = (val - stats_d[name]['median']) / mad
                    else:  # percentile
                        all_vals = data_dict[name]
                        z = (np.sum(all_vals <= val) / len(all_vals)) * 100
                    z_vals.append(z)
        if len(z_vals) >= 2:
            results.append(np.mean(z_vals))
        else:
            results.append(np.nan)
    return np.array(results)

print('\n=== 计算四种方案标准分 ===')
scores_z = compute_zscore(data, expert_stats, 'classic')
scores_robust = compute_zscore(data, expert_stats, 'robust')
scores_pct = compute_zscore(data, expert_stats, 'percentile')

# 方案D: 协商一致性加权Z-score（新模型）
# 权重 = 1/(1+CV), CV越小（评分越稳定）权重越大
weights = {name: 1.0/(1 + stats_d['std']/abs(stats_d['mean'])) 
           for name, stats_d in expert_stats.items() if abs(stats_d['mean'])>1e-10}
print(f'\n专家权重(CV倒数): {weights}')

scores_consensus = []
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
    scores_consensus.append(z_sum/w_sum if w_sum > 1e-10 else np.nan)
scores_consensus = np.array(scores_consensus)

# ===== 3. 验证指标 =====
print('\n=== 方案对比验证指标 ===')
valid = ~np.isnan(scores_z)

def show_metrics(scores, name, orig=None):
    v = scores[valid]
    ks_p = stats.kstest(v, 'norm')[1] if len(v)>10 else np.nan
    if orig is not None and len(v)>10:
        rho, _ = stats.spearmanr(orig[valid], v)
    else:
        rho = np.nan
    print(f"{name:25s} | μ={np.mean(v):+.3f} σ={np.std(v):.3f} KS-p={ks_p:.3f}{'✓' if ks_p>0.05 else '✗'} ρ={rho:.3f}")

# 原始总分（简单平均）
orig_scores = []
for row_idx in range(2, len(df)):
    vals = [pd.to_numeric(df.iloc[row_idx, c], errors='coerce') for c in expert_score_cols]
    vals = [v for v in vals if pd.notna(v)]
    orig_scores.append(np.mean(vals) if len(vals)>=2 else np.nan)
orig_scores = np.array(orig_scores)

show_metrics(orig_scores[valid], 'Original(raw mean)', orig_scores)
show_metrics(scores_z, 'A: Classic Z-score', orig_scores)
show_metrics(scores_robust, 'B: Robust Z-score', orig_scores)
show_metrics(scores_pct/100, 'C: Percentile/100', orig_scores)
show_metrics(scores_consensus, 'D: Consensus-Weighted(新)', orig_scores)

# ===== 4. 排名稳定性 =====
print('\n=== Top-20排名重合度(vs原始) ===')
top20_orig = set(np.argsort(-orig_scores[valid])[:20])
for name, scores in [('A:Z-score',scores_z),('B:Robust',scores_robust),('C:Percentile',scores_pct),('D:Consensus(新)',scores_consensus)]:
    top20 = set(np.argsort(-scores[valid])[:20])
    overlap = len(top20_orig & top20)
    print(f"{name:25s}: {overlap}/20 ({overlap*5:.0f}%)")

# ===== 5. 新模型公式 =====
print('\n=== 新标准分模型(方案D)公式 ===')
print('S_i = Σ_j [ w_j × (x_ij - μ_j)/σ_j ] / Σ_j w_j')
print('其中: w_j = 1 / (1 + CV_j), CV_j = σ_j/|μ_j| (变异系数)')
print('优势: 评分越稳定的专家权重越高，融合协商一致性思想')

# ===== 6. 输出结果 =====
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results', exist_ok=True)
results_df = pd.DataFrame({
    'work_idx': range(len(df)-2),
    'original': orig_scores,
    'zscore': scores_z,
    'robust': scores_robust, 
    'percentile': scores_pct,
    'consensus': scores_consensus,
    'rank_orig': pd.Series(orig_scores).rank(ascending=False),
    'rank_consensus': pd.Series(scores_consensus).rank(ascending=False)
})
results_df.to_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_final_results.csv', index=False, encoding='utf-8-sig')

with open(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q2_model_formula.txt', 'w', encoding='utf-8') as f:
    f.write("新标准分模型: S_i = Σ_j[w_j×(x_ij-μ_j)/σ_j]/Σ_j w_j\n")
    f.write(f"专家权重(CV倒数): {weights}\n")
    v = scores_consensus[valid]
    f.write(f"验证: KS-p={stats.kstest(v, \"norm\")[1]:.3f}, Spearman ρ={stats.spearmanr(orig_scores[valid], v)[0]:.3f}\n")

print('\n✓ 输出: results/q2_final_results.csv, q2_model_formula.txt')
print('\n✅ 分析完成！')
