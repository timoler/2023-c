import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# 中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 参数
# =========================================================
N_WORKS = 3000       # 作品数
N_EXPERTS = 125      # 专家数
K = 5                # 每篇作品需要的专家数

TOTAL_TASKS = N_WORKS * K
TARGET_LOAD = TOTAL_TASKS // N_EXPERTS

# 理论平均交叉次数
N_PAIRS = N_EXPERTS * (N_EXPERTS - 1) // 2
TOTAL_PAIR_OCCURRENCES = N_WORKS * (K * (K - 1) // 2)
TARGET_CROSS = TOTAL_PAIR_OCCURRENCES / N_PAIRS

print("=" * 60)
print("基本参数")
print("=" * 60)
print(f"作品数: {N_WORKS}")
print(f"专家数: {N_EXPERTS}")
print(f"每篇作品专家数: {K}")
print(f"总评审任务数: {TOTAL_TASKS}")
print(f"理论每位专家工作量: {TARGET_LOAD}")
print(f"专家对数量: {N_PAIRS}")
print(f"理论平均交叉次数: {TARGET_CROSS:.4f}")
print()


# =========================================================
# 2. 初始化
# =========================================================

# expert_load[i]：
# 第 i 位专家目前已经分配了多少篇作品
expert_load = np.zeros(N_EXPERTS, dtype=int)

# cross_matrix[i, j]：
# 专家 i 和专家 j 已经共同评审过多少篇
cross_matrix = np.zeros((N_EXPERTS, N_EXPERTS), dtype=int)

# 保存每一篇作品的专家
assignments = []

# 固定随机种子，方便复现
rng = np.random.default_rng(42)


# =========================================================
# 3. 计算候选专家的代价
# =========================================================

def expert_cost(candidate, selected):
    """
    candidate:
        待选择专家

    selected:
        当前这篇作品已经选中的专家

    目标：
    1. 工作量低的专家优先
    2. 与当前已选专家交叉次数较少的专家优先
    """

    # -----------------------------------------
    # 工作量代价
    # -----------------------------------------
    load_cost = expert_load[candidate] / TARGET_LOAD

    # -----------------------------------------
    # 交叉代价
    # -----------------------------------------
    if len(selected) == 0:
        cross_cost = 0
    else:
        cross_values = [
            cross_matrix[candidate, e]
            for e in selected
        ]

        # 越是已经频繁合作，代价越大
        cross_cost = np.mean(cross_values) / TARGET_CROSS

    # -----------------------------------------
    # 综合代价
    # -----------------------------------------
    # 交叉均衡比工作量稍微更重要
    cost = 1.0 * load_cost + 2.0 * cross_cost

    return cost


# =========================================================
# 4. 贪心分配
# =========================================================

print("开始进行贪心交叉分发……")

for work_id in range(N_WORKS):

    selected = []

    for _ in range(K):

        # 未选入当前作品、并且尚未达到120篇的专家
        candidates = [
            e for e in range(N_EXPERTS)
            if e not in selected
            and expert_load[e] < TARGET_LOAD
        ]

        # 计算每个候选专家代价
        costs = np.array([
            expert_cost(e, selected)
            for e in candidates
        ])

        # 找到最小代价
        min_cost = np.min(costs)

        # 允许浮点误差
        best_candidates = [
            candidates[i]
            for i in range(len(candidates))
            if abs(costs[i] - min_cost) < 1e-10
        ]

        # 如果多个专家同样优秀，随机选择
        chosen = rng.choice(best_candidates)

        selected.append(chosen)

    # 保存
    assignments.append(selected)

    # 更新专家工作量
    for e in selected:
        expert_load[e] += 1

    # 更新专家之间的交叉次数
    for e1, e2 in combinations(selected, 2):
        cross_matrix[e1, e2] += 1
        cross_matrix[e2, e1] += 1

    if (work_id + 1) % 500 == 0:
        print(f"已完成 {work_id + 1}/{N_WORKS} 篇")


print("\n分配完成！")


# =========================================================
# 5. 保存作品-专家分配结果
# =========================================================

assignment_df = pd.DataFrame(
    assignments,
    columns=[
        "专家1",
        "专家2",
        "专家3",
        "专家4",
        "专家5"
    ]
)

# 专家编号改成1~125
assignment_df = assignment_df + 1

assignment_df.insert(
    0,
    "作品编号",
    np.arange(1, N_WORKS + 1)
)

assignment_df.to_csv(
    "作品专家分配方案.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 6. 工作量评价
# =========================================================

print("\n" + "=" * 60)
print("专家工作量评价")
print("=" * 60)

print("平均工作量:", expert_load.mean())
print("最小工作量:", expert_load.min())
print("最大工作量:", expert_load.max())
print("工作量标准差:", expert_load.std())


load_df = pd.DataFrame({
    "专家编号": np.arange(1, N_EXPERTS + 1),
    "评审作品数": expert_load
})

load_df.to_csv(
    "专家工作量.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 7. 专家对交叉次数
# =========================================================

pair_records = []

cross_values = []

for i in range(N_EXPERTS):
    for j in range(i + 1, N_EXPERTS):

        value = cross_matrix[i, j]

        cross_values.append(value)

        pair_records.append({
            "专家A": i + 1,
            "专家B": j + 1,
            "共同评审作品数": value
        })

cross_values = np.array(cross_values)

pair_df = pd.DataFrame(pair_records)

pair_df.to_csv(
    "专家交叉次数.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. 评价指标
# =========================================================

print("\n" + "=" * 60)
print("专家交叉评价")
print("=" * 60)

print(f"理论平均交叉次数: {TARGET_CROSS:.4f}")
print(f"实际平均交叉次数: {cross_values.mean():.4f}")

print(f"最小交叉次数: {cross_values.min()}")
print(f"最大交叉次数: {cross_values.max()}")

print(f"交叉次数标准差: {cross_values.std():.4f}")
print(f"交叉次数方差: {cross_values.var():.4f}")

print(
    f"零交叉专家对比例: "
    f"{np.mean(cross_values == 0) * 100:.4f}%"
)

print(
    f"交叉3~4次专家对比例: "
    f"{np.mean((cross_values >= 3) & (cross_values <= 4)) * 100:.4f}%"
)

print(
    f"交叉2~5次专家对比例: "
    f"{np.mean((cross_values >= 2) & (cross_values <= 5)) * 100:.4f}%"
)


# =========================================================
# 9. 输出交叉次数频数
# =========================================================

counter = Counter(cross_values)

print("\n专家对共同评审次数分布：")

for value in sorted(counter.keys()):
    print(
        f"共同评审 {value} 次："
        f"{counter[value]} 对专家 "
        f"({counter[value] / N_PAIRS * 100:.2f}%)"
    )


distribution_df = pd.DataFrame({
    "共同评审次数": list(sorted(counter.keys())),
    "专家对数量": [
        counter[k]
        for k in sorted(counter.keys())
    ]
})

distribution_df["比例"] = (
    distribution_df["专家对数量"]
    / N_PAIRS
)

distribution_df.to_csv(
    "专家交叉次数分布.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. 目标函数
# =========================================================

objective = np.sum(
    (cross_values - TARGET_CROSS) ** 2
)

mse_cross = np.mean(
    (cross_values - TARGET_CROSS) ** 2
)

print("\n" + "=" * 60)
print("优化目标")
print("=" * 60)

print(f"交叉偏差平方和 SSE = {objective:.4f}")
print(f"交叉均方误差 MSE = {mse_cross:.6f}")


# =========================================================
# 11. 图1：专家工作量
# =========================================================

plt.figure(figsize=(12, 5))

plt.bar(
    np.arange(1, N_EXPERTS + 1),
    expert_load
)

plt.axhline(
    TARGET_LOAD,
    linestyle="--",
    label=f"理论工作量 = {TARGET_LOAD}"
)

plt.xlabel("专家编号")
plt.ylabel("评审作品数量")
plt.title("125位专家评审工作量分布")
plt.legend()

plt.tight_layout()

plt.savefig(
    "专家工作量分布.png",
    dpi=300
)

plt.show()


# =========================================================
# 12. 图2：专家对交叉次数直方图
# =========================================================

plt.figure(figsize=(9, 5))

bins = np.arange(
    cross_values.min() - 0.5,
    cross_values.max() + 1.5,
    1
)

plt.hist(
    cross_values,
    bins=bins,
    edgecolor="black"
)

plt.axvline(
    TARGET_CROSS,
    linestyle="--",
    label=f"理论均值={TARGET_CROSS:.3f}"
)

plt.xlabel("任意两位专家共同评审作品数")
plt.ylabel("专家对数量")
plt.title("专家两两交叉评审次数分布")

plt.legend()
plt.tight_layout()

plt.savefig(
    "专家交叉次数分布.png",
    dpi=300
)

plt.show()


# =========================================================
# 13. 图3：交叉矩阵热力图
# =========================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    cross_matrix,
    aspect="auto"
)

plt.colorbar(
    label="共同评审作品数"
)

plt.xlabel("专家编号")
plt.ylabel("专家编号")
plt.title("专家交叉评审矩阵")

plt.tight_layout()

plt.savefig(
    "专家交叉矩阵热力图.png",
    dpi=300
)

plt.show()


print("\n全部结果已经保存。")
