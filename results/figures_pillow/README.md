# [PLOT] C题第二问：Pillow方案可视化图表

> [OK] 本方案绕过matplotlib渲染问题，100%兼容当前环境
> [DIR] 输出目录: `figures_pillow/` | 格式: PNG | 分辨率: 95% quality

| 图号 | 文件名 | 说明 | 建议论文位置 |
|------|---------|------|-------------|
| 图1 | `q2_expert_distribution.png` | 专家原始评分箱线图 | 2.1 数据描述 |
| 图2 | `q2_zscore_dist.png` | Z-score标准分直方图 | 2.3 方法验证 |
| 图3 | `q2_correlation_scatter.png` | Consensus vs 原始相关性 | 2.3 保序性 |
| 图4 | `q2_top50_heatmap.png` | Top-50排名热力图 | 2.4 结果展示 |
| 图5 | `q2_expert_weights.png` | 专家权重条形图 | 2.2 模型设计 |
| 图6 | `q2_method_radar.png` | 多维指标雷达图 | 2.5 方案对比 |

## [FIX] 使用说明
1. 所有图表可直接插入Word/LaTeX论文
2. 如需更高分辨率，修改`PillowChart.save()`的quality参数
3. 中文字体问题: 修改脚本顶部`FONT_CANDIDATES`列表
