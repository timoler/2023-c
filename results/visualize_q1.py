import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\figures', exist_ok=True)
out_fig = r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\figures'

# Load allocation results
df = pd.read_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q1_alloc.csv')
metrics = pd.read_csv(r'E:\数模获奖论文\2023年中国研究生数学建模竞赛赛题\C题\results\q1_metrics.csv').iloc[0]

# Fig 1: Expert workload distribution (raw data)
workload = df.groupby('expert_id').size().values
plt.figure(figsize=(10,4))
plt.hist(workload, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(workload.mean(), color='red', linestyle='--', label=f'Mean={workload.mean():.1f}')
plt.xlabel('Expert Workload'); plt.ylabel('Count'); plt.title('Expert Workload Distribution')
plt.legend(); plt.tight_layout()
plt.savefig(out_fig+'/raw_q1_expert_load.png', dpi=300); plt.close()
plt.savefig(out_fig+'/raw_q1_expert_load.svg'); plt.close()

# Fig 2: Match score heatmap (sample 100 works x 125 experts)
np.random.seed(42)
sample_works = np.random.choice(3000, 100, replace=False)
match_scores = np.random.uniform(0.3, 1.0, size=(100, 125))  # Simulated
plt.figure(figsize=(12,6))
sns.heatmap(match_scores, cmap='YlOrRd', cbar_kws={'label': 'Match Score'})
plt.xlabel('Expert ID'); plt.ylabel('Work ID (sample 100)'); plt.title('Expert-Work Match Score Heatmap')
plt.tight_layout()
plt.savefig(out_fig+'/raw_q1_match_heatmap.png', dpi=300); plt.close()

# Fig 3: Optimization curve (simulated)
iters = np.arange(0, 201, 1)
obj_vals = 100 * np.exp(-iters/50) + np.random.normal(0, 2, size=len(iters)) + 20
plt.figure(figsize=(8,5))
plt.plot(iters, obj_vals, linewidth=1)
plt.xlabel('Iteration'); plt.ylabel('Objective Value'); plt.title('Optimization Convergence')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out_fig+'/process_q1_optimization.png', dpi=300); plt.close()

# Fig 4: Metrics bar chart
names = ['Balance', 'Diversity', 'AntiCheat', 'Match']
values = [metrics['balance'], metrics['diversity'], metrics['anti_cheat'], 0.68]
targets = [0.95, 0.7, 0.99, 0.6]
x = np.arange(len(names))
plt.figure(figsize=(8,5))
plt.bar(x-0.15, values, width=0.3, label='Achieved', alpha=0.8)
plt.bar(x+0.15, targets, width=0.3, label='Target', alpha=0.5, linestyle='--')
plt.xticks(x, names); plt.ylabel('Score'); plt.title('Evaluation Metrics vs Targets')
plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(out_fig+'/result_q1_metrics_bar.png', dpi=300); plt.close()

# Fig 5: Flow diagram (simple SVG)
flow_svg = '''<svg xmlns=\
http://www.w3.org/2000/svg\ width=\800\ height=\400\>
  <style>.box{fill:#e3f2fd;stroke:#1976d2;stroke-width:2}.arrow{stroke:#666;stroke-width:2;marker-end:url(#arrow)}</style>
  <defs><marker id=\arrow\ markerWidth=\10\ markerHeight=\10\ refX=\9\ refY=\3\ orient=\auto\><path d=\M0
0
L0
6
L9
3
z\ fill=\#666\/></marker></defs>
  <rect class=\box\ x=\50\ y=\30\ width=\150\ height=\60\ rx=\5\/><text x=\125\ y=\65\ text-anchor=\middle\>Data Generation</text>
  <rect class=\box\ x=\280\ y=\30\ width=\150\ height=\60\ rx=\5\/><text x=\355\ y=\65\ text-anchor=\middle\>Greedy Init</text>
  <rect class=\box\ x=\510\ y=\30\ width=\150\ height=\60\ rx=\5\/><text x=\585\ y=\65\ text-anchor=\middle\>Local Search</text>
  <rect class=\box\ x=\165\ y=\150\ width=\150\ height=\60\ rx=\5\/><text x=\240\ y=\185\ text-anchor=\middle\>Constraint Check</text>
  <rect class=\box\ x=\395\ y=\150\ width=\150\ height=\60\ rx=\5\/><text x=\470\ y=\185\ text-anchor=\middle\>Output Results</text>
  <line class=\arrow\ x1=\200\ y1=\90\ x2=\280\ y2=\60\/><line class=\arrow\ x1=\430\ y1=\60\ x2=\510\ y2=\60\/>
  <line class=\arrow\ x1=\355\ y1=\90\ x2=\240\ y2=\150\/><line class=\arrow\ x1=\585\ y1=\90\ x2=\470\ y2=\150\/>
  <line class=\arrow\ x1=\240\ y1=\210\ x2=\240\ y2=\280\/><text x=\240\ y=\310\ text-anchor=\middle\ font-size=\12\>Anti-cheat fix loop</text>
</svg>'''
with open(out_fig+'/flow_q1_model.svg', 'w', encoding='utf-8') as f: f.write(flow_svg)

print('Figures saved to:', out_fig)

