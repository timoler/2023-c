import numpy as np
import pandas as pd
from itertools import combinations

# ============================================================
# 参数
# ============================================================
N_WORKS = 3000
N_EXPERTS = 125
K = 5
TARGET_LOAD = 120

N_PAIRS = N_EXPERTS * (N_EXPERTS - 1) // 2
TOTAL_PAIR_CROSS = N_WORKS * (K * (K - 1) // 2)
TARGET_CROSS = TOTAL_PAIR_CROSS / N_PAIRS

N_RUNS = 100

print("=" * 60)
print("随机均衡分配基线实验")
print("=" * 60)
print(f"作品数: {N_WORKS}")
print(f"专家数: {N_EXPERTS}")
print(f"每篇专家数: {K}")
print(f"每位专家任务量: {TARGET_LOAD}")
print(f"专家对总数: {N_PAIRS}")
print(f"理论平均交叉次数: {TARGET_CROSS:.4f}")
print(f"随机实验次数: {N_RUNS}")
print()


# ============================================================
# 随机均衡分配
# ============================================================
def random_balanced_assignment(seed):

    rng = np.random.default_rng(seed)

    # 逐篇构造：每篇随机抽 5 位不同专家，
    # 抽样概率正比于该专家剩余任务量（保证工作量始终均衡、最后恰好用完）。
    #
    # 注：原「打乱整个专家池再检查每篇无重复」的思路在 N=3000 时
    # 成功概率约为 0.92^3000 ≈ 0，10000 次尝试也几乎不可能成功，
    # 因此这里直接使用逐篇构造（这才是真正生效的随机基线）。
    remaining = np.full(
        N_EXPERTS,
        TARGET_LOAD,
        dtype=int
    )

    assignments = []

    for work in range(N_WORKS):

        works_left = N_WORKS - work  # 含当前这篇，剩余还需填的作品数

        # 必选：剩余任务数 == 剩余作品数 的专家
        # （否则他后面每个作品都必须出现，会突破 K 限制，导致无解）
        forced = np.where(
            remaining >= works_left
        )[0]

        # 其余名额从「还有余量但未到必选线」的专家里随机抽，按剩余量加权
        others = np.where(
            (remaining > 0) & (remaining < works_left)
        )[0]

        need = K - len(forced)

        if need > 0:
            probabilities = remaining[others].astype(float)
            probabilities /= probabilities.sum()

            extra = rng.choice(
                others,
                size=need,
                replace=False,
                p=probabilities
            )
        else:
            extra = np.array([], dtype=int)

        chosen = np.concatenate([forced, extra])

        assignments.append(chosen)

        remaining[chosen] -= 1

    assignments = np.array(assignments)

    # 理论上应该全部用完
    if remaining.sum() != 0:
        raise RuntimeError(
            "随机均衡分配未完全满足工作量约束"
        )

    return assignments


# ============================================================
# 评价函数
# ============================================================
def evaluate(assignments):

    expert_load = np.zeros(
        N_EXPERTS,
        dtype=int
    )

    cross_matrix = np.zeros(
        (N_EXPERTS, N_EXPERTS),
        dtype=int
    )

    for row in assignments:

        # 工作量
        expert_load[row] += 1

        # 专家两两交叉
        for a, b in combinations(row, 2):
            cross_matrix[a, b] += 1
            cross_matrix[b, a] += 1

    cross_values = []

    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            cross_values.append(
                cross_matrix[i, j]
            )

    cross_values = np.array(cross_values)

    result = {
        "load_mean": expert_load.mean(),
        "load_std": expert_load.std(),
        "cross_mean": cross_values.mean(),
        "cross_std": cross_values.std(),
        "cross_min": cross_values.min(),
        "cross_max": cross_values.max(),
        "zero_ratio": np.mean(
            cross_values == 0
        ),
        "ratio_3_4": np.mean(
            (cross_values >= 3)
            & (cross_values <= 4)
        ),
        "ratio_2_5": np.mean(
            (cross_values >= 2)
            & (cross_values <= 5)
        ),
        "mse": np.mean(
            (cross_values - TARGET_CROSS) ** 2
        )
    }

    return result


# ============================================================
# 跑100次
# ============================================================
results = []

for run in range(N_RUNS):

    seed = 1000 + run

    assignments = random_balanced_assignment(
        seed
    )

    result = evaluate(assignments)

    result["run"] = run + 1
    result["seed"] = seed

    results.append(result)

    if (run + 1) % 10 == 0:
        print(
            f"完成 {run + 1}/{N_RUNS}"
        )


df = pd.DataFrame(results)

df.to_csv(
    "问题一/results/q1_random_baseline_100runs.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("100次随机均衡分配统计结果")
print("=" * 60)

metrics = [
    "cross_std",
    "cross_min",
    "cross_max",
    "zero_ratio",
    "ratio_3_4",
    "ratio_2_5",
    "mse"
]

for metric in metrics:

    print(f"\n{metric}")

    print(
        f"平均值 = "
        f"{df[metric].mean():.6f}"
    )

    print(
        f"标准差 = "
        f"{df[metric].std():.6f}"
    )

    print(
        f"最小值 = "
        f"{df[metric].min():.6f}"
    )

    print(
        f"最大值 = "
        f"{df[metric].max():.6f}"
    )


# ============================================================
# 与贪心结果比较
# ============================================================

# 从 q1_solve.py 生成的结果读取贪心方案指标，避免硬编码展示值。
try:
    greedy_cross = pd.read_csv(
        "问题一/results/q1_cross_counts.csv",
        encoding="utf-8-sig"
    )["共同评审作品数"].to_numpy()

    greedy = {
        "cross_std": float(greedy_cross.std()),
        "cross_min": int(greedy_cross.min()),
        "cross_max": int(greedy_cross.max()),
        "zero_ratio": float((greedy_cross == 0).mean()),
        "ratio_3_4": float(
            ((greedy_cross >= 3) & (greedy_cross <= 4)).mean()
        ),
        "ratio_2_5": float(
            ((greedy_cross >= 2) & (greedy_cross <= 5)).mean()
        ),
        "mse": float(((greedy_cross - TARGET_CROSS) ** 2).mean()),
    }
except FileNotFoundError:
    raise SystemExit(
        "未找到 问题一/results/q1_cross_counts.csv，"
        "请先运行 q1_solve.py 生成贪心结果。"
    )

comparison = []

for metric in metrics:

    comparison.append({
        "指标": metric,
        "随机分配100次均值": df[metric].mean(),
        "均衡贪心": greedy[metric]
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    "问题一/results/q1_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("随机分配 vs 均衡贪心")
print("=" * 60)

print(comparison_df.to_string(index=False))

print("\n实验完成。")
