#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C题第二问：按专家编码计算标准分、数据1校准权重并生成可复现结果。

修正点：
1. 不再把“专家一/二/三...”位置列当成同一位专家。
2. 第二阶段只读取“专家编码-原始分”配对，不混用标准分列。
3. 用数据1一等奖协商排序校准专家权重，排名方向按“名次越小越好”处理。
4. 验证指标、鲁棒性和运行时间均由代码真实计算。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA21_PATH = PROJECT_ROOT / "数据2.1 .xlsx"
DATA1_PATH = PROJECT_ROOT / "数据1.xlsx"
RESULT_DIR = PROJECT_ROOT / "results" / "q2_corrected"
FIG_DIR = PROJECT_ROOT / "问题二" / "figures_corrected"

PHASE1_GROUPS = [
    ("phase1", "expert_1", 5, 6),
    ("phase1", "expert_2", 8, 9),
    ("phase1", "expert_3", 11, 12),
    ("phase1", "expert_4", 14, 15),
    ("phase1", "expert_5", 17, 18),
]
PHASE2_GROUPS = [
    ("phase2", "expert_1", 23, 24),
    ("phase2", "expert_2", 27, 28),
    ("phase2", "expert_3", 31, 32),
]
ALL_GROUPS = PHASE1_GROUPS + PHASE2_GROUPS
RNG_SEED = 20230921


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def spearman_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    rx = df["x"].rank(method="average").to_numpy(dtype=float)
    ry = df["y"].rank(method="average").to_numpy(dtype=float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom > 1e-12 else np.nan


def read_raw_excel(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{path.name} 行数异常：期望 {expected_rows}，实际 {len(df)}")
    if df.shape[1] < 35:
        raise ValueError(f"{path.name} 列数异常：至少需要35列，实际 {df.shape[1]}")
    if pd.to_numeric(df.iloc[3, 0], errors="coerce") is pd.NA or pd.isna(pd.to_numeric(df.iloc[3, 0], errors="coerce")):
        raise ValueError(f"{path.name} 第4行第1列不是作品成绩，表头结构可能变化")
    return df


def extract_scores(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    works: list[dict] = []

    for ridx in range(3, len(df)):
        work_no = ridx - 2
        final_score = pd.to_numeric(df.iloc[ridx, 0], errors="coerce")
        consensus_rank = pd.to_numeric(df.iloc[ridx, 1], errors="coerce")
        prize = df.iloc[ridx, 2]
        if pd.isna(final_score) and pd.isna(consensus_rank):
            continue

        works.append(
            {
                "source": source_name,
                "work_no": work_no,
                "row_index": ridx,
                "final_score": float(final_score) if pd.notna(final_score) else np.nan,
                "consensus_rank": float(consensus_rank) if pd.notna(consensus_rank) else np.nan,
                "prize": prize,
            }
        )

        for stage, slot, expert_col, raw_col in ALL_GROUPS:
            expert = df.iloc[ridx, expert_col]
            raw_score = pd.to_numeric(df.iloc[ridx, raw_col], errors="coerce")
            if pd.isna(expert) or pd.isna(raw_score):
                continue
            records.append(
                {
                    "source": source_name,
                    "work_no": work_no,
                    "row_index": ridx,
                    "stage": stage,
                    "slot": slot,
                    "expert_code": str(expert).strip(),
                    "raw_score": float(raw_score),
                    "final_score": float(final_score) if pd.notna(final_score) else np.nan,
                    "consensus_rank": float(consensus_rank) if pd.notna(consensus_rank) else np.nan,
                    "prize": prize,
                }
            )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(works)


def expert_statistics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (stage, expert), g in long_df.groupby(["stage", "expert_code"], sort=True):
        scores = g["raw_score"].to_numpy(dtype=float)
        mean = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)) * 1.4826)
        rows.append(
            {
                "stage": stage,
                "expert_code": expert,
                "n": int(len(scores)),
                "mean": mean,
                "std": sigma,
                "median": median,
                "mad": mad,
                "cv": sigma / abs(mean) if abs(mean) > 1e-12 else np.nan,
                "raw_min": float(np.min(scores)),
                "raw_max": float(np.max(scores)),
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df["cv_weight"] = 1.0 / (1.0 + stats_df["cv"].fillna(0.0))
    return stats_df


def add_standardized_scores(long_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(stats_df, on=["stage", "expert_code"], how="left", validate="many_to_one")
    merged["z_score"] = np.where(
        merged["std"] > 1e-12,
        (merged["raw_score"] - merged["mean"]) / merged["std"],
        0.0,
    )
    merged["robust_z"] = np.where(
        merged["mad"] > 1e-12,
        (merged["raw_score"] - merged["median"]) / merged["mad"],
        0.0,
    )

    merged["percentile"] = np.nan
    for (_stage, _expert), idx in merged.groupby(["stage", "expert_code"]).groups.items():
        merged.loc[idx, "percentile"] = merged.loc[idx, "raw_score"].rank(method="average", pct=True).to_numpy(dtype=float)
    return merged


def calibration_from_data1(data1_long: pd.DataFrame) -> pd.DataFrame:
    # 一等奖“名次越小越好”；因此用 -rank 作为质量方向，避免把相关性方向反过来。
    first_prize = data1_long[data1_long["prize"].astype(str).str.contains("一等奖", na=False)].copy()
    rows: list[dict] = []
    for (stage, expert), g in first_prize.groupby(["stage", "expert_code"], sort=True):
        valid = g[["raw_score", "consensus_rank"]].dropna()
        if len(valid) < 3 or valid["raw_score"].nunique() < 2 or valid["consensus_rank"].nunique() < 2:
            corr = np.nan
        else:
            corr = spearman_corr(valid["raw_score"], -valid["consensus_rank"])
        calibration = 0.5 if pd.isna(corr) else float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
        shrink = len(valid) / (len(valid) + 10.0)
        rows.append(
            {
                "stage": stage,
                "expert_code": expert,
                "calibration_n": int(len(valid)),
                "calibration_spearman_score_vs_negative_rank": float(corr) if pd.notna(corr) else np.nan,
                "calibration_p_value": np.nan,
                "calibration_weight": float(0.5 + shrink * (calibration - 0.5)),
            }
        )
    return pd.DataFrame(rows)


def attach_weights(scored_long: pd.DataFrame, calibration_df: pd.DataFrame) -> pd.DataFrame:
    weighted = scored_long.merge(calibration_df, on=["stage", "expert_code"], how="left")
    weighted["calibration_weight"] = weighted["calibration_weight"].fillna(0.75)
    weighted["final_weight"] = weighted["cv_weight"] * weighted["calibration_weight"]
    return weighted


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask].to_numpy(dtype=float), weights=weights[mask].to_numpy(dtype=float)))


def aggregate_work_scores(scored_long: pd.DataFrame, works_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for work_no, g in scored_long.groupby("work_no", sort=True):
        rows.append(
            {
                "work_no": int(work_no),
                "raw_mean": float(g["raw_score"].mean()),
                "classic_z": float(g["z_score"].mean()),
                "robust_z": float(g["robust_z"].mean()),
                "percentile": float(g["percentile"].mean()),
                "cv_weighted_z": weighted_average(g["z_score"], g["cv_weight"]),
                "calibrated_weighted_z": weighted_average(g["z_score"], g["final_weight"]),
                "review_count": int(len(g)),
                "expert_count": int(g["expert_code"].nunique()),
            }
        )
    result = works_df.merge(pd.DataFrame(rows), on="work_no", how="left")
    for col in ["raw_mean", "classic_z", "robust_z", "percentile", "cv_weighted_z", "calibrated_weighted_z"]:
        result[f"{col}_rank"] = result[col].rank(ascending=False, method="min")
    return result


def standardize_vector(s: pd.Series) -> np.ndarray:
    arr = s.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return arr
    sd = np.std(arr, ddof=1)
    return np.zeros_like(arr) if sd <= 1e-12 else (arr - np.mean(arr)) / sd


def validation_metrics(result_df: pd.DataFrame, elapsed_seconds: float) -> pd.DataFrame:
    methods = ["classic_z", "robust_z", "percentile", "cv_weighted_z", "calibrated_weighted_z"]
    rows: list[dict] = []
    for method in methods:
        valid = result_df[["raw_mean", "final_score", method]].dropna()
        zed = standardize_vector(valid[method])
        ks_p = stats.kstest(zed, "norm").pvalue if len(zed) else np.nan
        rho_raw = spearman_corr(valid["raw_mean"], valid[method]) if len(valid) > 2 else np.nan
        rho_final = spearman_corr(valid["final_score"], valid[method]) if len(valid) > 2 else np.nan
        rows.append(
            {
                "method": method,
                "n_valid": int(len(valid)),
                "ks_p_after_rescale": float(ks_p) if pd.notna(ks_p) else np.nan,
                "spearman_vs_raw_mean": float(rho_raw) if pd.notna(rho_raw) else np.nan,
                "spearman_vs_original_final_score": float(rho_final) if pd.notna(rho_final) else np.nan,
                "milliseconds_per_work": elapsed_seconds * 1000.0 / max(len(result_df), 1),
            }
        )
    return pd.DataFrame(rows)


def robustness_metrics(scored_long: pd.DataFrame, works_df: pd.DataFrame, repeats: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    base = aggregate_work_scores(scored_long, works_df)
    methods = ["classic_z", "robust_z", "percentile", "cv_weighted_z", "calibrated_weighted_z"]
    base_top = {m: set(base.nlargest(50, m)["work_no"].astype(int)) for m in methods}
    rows = {m: [] for m in methods}

    for _ in range(repeats):
        perturbed = scored_long.copy()
        hit = rng.random(len(perturbed)) < 0.05
        noise = rng.normal(0.0, 8.0, size=len(perturbed))
        perturbed.loc[hit, "raw_score"] = np.clip(perturbed.loc[hit, "raw_score"] + noise[hit], 0, 100)
        perturbed["z_score"] = np.where(
            perturbed["std"] > 1e-12,
            (perturbed["raw_score"] - perturbed["mean"]) / perturbed["std"],
            0.0,
        )
        perturbed["robust_z"] = np.where(
            perturbed["mad"] > 1e-12,
            (perturbed["raw_score"] - perturbed["median"]) / perturbed["mad"],
            0.0,
        )
        for (_stage, _expert), idx in perturbed.groupby(["stage", "expert_code"]).groups.items():
            perturbed.loc[idx, "percentile"] = perturbed.loc[idx, "raw_score"].rank(method="average", pct=True)

        current = aggregate_work_scores(perturbed, works_df)
        for method in methods:
            cur_top = set(current.nlargest(50, method)["work_no"].astype(int))
            rows[method].append(1.0 - len(base_top[method] & cur_top) / 50.0)

    return pd.DataFrame(
        {
            "method": method,
            "top50_change_rate_mean": float(np.mean(values)),
            "top50_change_rate_p95": float(np.percentile(values, 95)),
            "perturbation_repeats": repeats,
            "perturbed_review_share": 0.05,
            "noise_sd": 8.0,
        }
        for method, values in rows.items()
    )


def make_figures(scored_long: pd.DataFrame, result_df: pd.DataFrame, stats_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    font = ImageFont.load_default()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    def canvas(title: str, w: int = 1200, h: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((w // 2, 24), title, fill="black", font=font, anchor="mt")
        return img, draw

    def sx(x: float, xmin: float, xmax: float, left: int, right: int) -> int:
        if abs(xmax - xmin) < 1e-12:
            return (left + right) // 2
        return int(left + (x - xmin) / (xmax - xmin) * (right - left))

    def sy(y: float, ymin: float, ymax: float, top: int, bottom: int) -> int:
        if abs(ymax - ymin) < 1e-12:
            return (top + bottom) // 2
        return int(bottom - (y - ymin) / (ymax - ymin) * (bottom - top))

    def save(img: Image.Image, name: str) -> None:
        img.save(FIG_DIR / name, "PNG")

    top_experts = stats_df.sort_values("n", ascending=False).head(12)
    img, draw = canvas("Score distribution by real expert code (top 12 by n)")
    left, right, top, bottom = 90, 1150, 90, 620
    draw.line([(left, bottom), (right, bottom)], fill="black")
    draw.line([(left, top), (left, bottom)], fill="black")
    all_scores = scored_long["raw_score"].to_numpy(dtype=float)
    ymin, ymax = float(np.nanmin(all_scores)), float(np.nanmax(all_scores))
    step = (right - left) / len(top_experts)
    for i, r in enumerate(top_experts.itertuples()):
        vals = scored_long.loc[(scored_long["stage"] == r.stage) & (scored_long["expert_code"] == r.expert_code), "raw_score"].to_numpy()
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = np.min(vals), np.max(vals)
        x = int(left + step * (i + 0.5))
        box_w = int(step * 0.45)
        draw.line([(x, sy(lo, ymin, ymax, top, bottom)), (x, sy(hi, ymin, ymax, top, bottom))], fill=colors[0], width=2)
        draw.rectangle(
            [(x - box_w // 2, sy(q3, ymin, ymax, top, bottom)), (x + box_w // 2, sy(q1, ymin, ymax, top, bottom))],
            outline="black",
            fill="#D9EAF7",
        )
        draw.line([(x - box_w // 2, sy(med, ymin, ymax, top, bottom)), (x + box_w // 2, sy(med, ymin, ymax, top, bottom))], fill=colors[1], width=2)
        draw.text((x, bottom + 12), str(r.stage), fill="black", font=font, anchor="mt")
        draw.text((x, bottom + 28), str(r.expert_code), fill="black", font=font, anchor="mt")
    draw.text((18, (top + bottom) // 2), "Raw score", fill="black", font=font)
    save(img, "raw_q2_expert_score_distribution.png")

    values = standardize_vector(result_df["calibrated_weighted_z"])
    counts, edges = np.histogram(values, bins=28)
    img, draw = canvas("Distribution of calibrated weighted scores", 1000, 620)
    left, right, top, bottom = 80, 950, 80, 520
    draw.line([(left, bottom), (right, bottom)], fill="black")
    draw.line([(left, top), (left, bottom)], fill="black")
    bar_w = (right - left) / len(counts)
    max_count = max(counts) if len(counts) else 1
    for i, count in enumerate(counts):
        x0 = int(left + i * bar_w)
        x1 = int(left + (i + 1) * bar_w - 2)
        y = sy(float(count), 0, float(max_count), top, bottom)
        draw.rectangle([(x0, y), (x1, bottom)], fill=colors[0])
    draw.text(((left + right) // 2, bottom + 36), "Calibrated weighted score (rescaled)", fill="black", font=font, anchor="mt")
    draw.text((18, (top + bottom) // 2), "Works", fill="black", font=font)
    save(img, "process_q2_calibrated_score_hist.png")

    valid = result_df[["raw_mean", "calibrated_weighted_z"]].dropna()
    rho = spearman_corr(valid["raw_mean"], valid["calibrated_weighted_z"])
    img, draw = canvas(f"Calibrated score vs raw mean (Spearman={rho:.4f})", 900, 720)
    left, right, top, bottom = 90, 840, 80, 620
    draw.line([(left, bottom), (right, bottom)], fill="black")
    draw.line([(left, top), (left, bottom)], fill="black")
    xmin, xmax = valid["raw_mean"].min(), valid["raw_mean"].max()
    ymin, ymax = valid["calibrated_weighted_z"].min(), valid["calibrated_weighted_z"].max()
    for row in valid.itertuples():
        x = sx(row.raw_mean, xmin, xmax, left, right)
        y = sy(row.calibrated_weighted_z, ymin, ymax, top, bottom)
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=colors[1])
    draw.text(((left + right) // 2, bottom + 36), "Raw mean score", fill="black", font=font, anchor="mt")
    draw.text((18, (top + bottom) // 2), "Calibrated weighted score", fill="black", font=font)
    save(img, "result_q2_score_vs_raw_mean.png")

    method_cols = ["raw_mean", "classic_z", "robust_z", "cv_weighted_z", "calibrated_weighted_z"]
    corr = pd.DataFrame(index=method_cols, columns=method_cols, dtype=float)
    for left_name in method_cols:
        for right_name in method_cols:
            corr.loc[left_name, right_name] = spearman_corr(result_df[left_name], result_df[right_name])
    img, draw = canvas("Spearman correlation among scoring methods", 980, 760)
    x0, y0, cell = 300, 100, 90
    for i, row_name in enumerate(method_cols):
        draw.text((x0 - 10, y0 + i * cell + cell // 2), row_name, fill="black", font=font, anchor="rm")
        draw.text((x0 + i * cell + cell // 2, y0 + len(method_cols) * cell + 15), row_name, fill="black", font=font, anchor="mt")
        for j, col_name in enumerate(method_cols):
            val = float(corr.loc[row_name, col_name])
            shade = int(255 - abs(val) * 120)
            fill = (shade, 235 if val >= 0 else shade, shade if val >= 0 else 235)
            draw.rectangle([(x0 + j * cell, y0 + i * cell), (x0 + (j + 1) * cell, y0 + (i + 1) * cell)], fill=fill, outline="white")
            draw.text((x0 + j * cell + cell // 2, y0 + i * cell + cell // 2), f"{val:.2f}", fill="black", font=font, anchor="mm")
    save(img, "process_q2_method_correlation_heatmap.png")

    weight_df = (
        scored_long[["stage", "expert_code", "cv_weight", "calibration_weight", "final_weight"]]
        .drop_duplicates()
        .sort_values("final_weight", ascending=False)
        .head(15)
    )
    img, draw = canvas("Expert weights after Data 1 calibration (top 15)", 1000, 760)
    left, right, top, bottom = 240, 940, 80, 680
    max_w = float(weight_df["final_weight"].max())
    bar_h = (bottom - top) / len(weight_df)
    for i, r in enumerate(weight_df.itertuples()):
        y_mid = int(top + bar_h * (i + 0.5))
        bar_len = int((r.final_weight / max_w) * (right - left))
        draw.text((left - 10, y_mid), f"{r.stage}-{r.expert_code}", fill="black", font=font, anchor="rm")
        draw.rectangle([(left, y_mid - 10), (left + bar_len, y_mid + 10)], fill=colors[2])
        draw.text((left + bar_len + 8, y_mid), f"{r.final_weight:.3f}", fill="black", font=font, anchor="lm")
    save(img, "result_q2_calibrated_expert_weights.png")

    plot_df = metrics_df.set_index("method")
    img, draw = canvas("Validation metrics by method", 1100, 700)
    left, right, top, bottom = 100, 1040, 90, 560
    draw.line([(left, bottom), (right, bottom)], fill="black")
    draw.line([(left, top), (left, bottom)], fill="black")
    group_w = (right - left) / len(plot_df)
    for i, (method, row) in enumerate(plot_df.iterrows()):
        gx = int(left + i * group_w + group_w * 0.2)
        b1 = sy(float(row["spearman_vs_original_final_score"]), 0, 1, top, bottom)
        b2 = sy(float(row["ks_p_after_rescale"]), 0, 1, top, bottom)
        draw.rectangle([(gx, b1), (gx + 22, bottom)], fill=colors[0])
        draw.rectangle([(gx + 28, b2), (gx + 50, bottom)], fill=colors[4])
        draw.text((gx + 25, bottom + 12), method, fill="black", font=font, anchor="mt")
    draw.text((left + 10, top + 10), "blue=Spearman vs final, yellow=KS p", fill="black", font=font)
    save(img, "result_q2_method_metrics.png")


def write_manifest(metrics_df: pd.DataFrame, robustness_df: pd.DataFrame) -> None:
    manifest = {
        "command": "python 问题二/q2_visualize.py",
        "random_seed": RNG_SEED,
        "inputs": {
            DATA21_PATH.name: file_sha256(DATA21_PATH),
            DATA1_PATH.name: file_sha256(DATA1_PATH),
        },
        "outputs": {
            "work_scores": str(RESULT_DIR / "q2_work_scores_corrected.csv"),
            "review_long": str(RESULT_DIR / "q2_review_long_corrected.csv"),
            "expert_stats": str(RESULT_DIR / "q2_expert_stats_corrected.csv"),
            "metrics": str(RESULT_DIR / "q2_validation_metrics.csv"),
            "robustness": str(RESULT_DIR / "q2_robustness_metrics.csv"),
            "figures": str(FIG_DIR),
        },
        "best_method": "calibrated_weighted_z",
        "validation_summary": metrics_df.to_dict(orient="records"),
        "robustness_summary": robustness_df.to_dict(orient="records"),
    }
    (RESULT_DIR / "复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("=" * 72)
    print("C题第二问：按专家编码修正版")
    print("=" * 72, flush=True)

    start = time.perf_counter()
    df21 = read_raw_excel(DATA21_PATH, expected_rows=888)
    df1 = read_raw_excel(DATA1_PATH, expected_rows=2018)
    print("[1/6] Excel读取完成", flush=True)
    data21_long, data21_works = extract_scores(df21, "data2.1")
    data1_long, _data1_works = extract_scores(df1, "data1")
    print("[2/6] 长表展开完成", flush=True)

    if len(data21_works) != 885:
        raise ValueError(f"数据2.1作品数异常：期望 885，实际 {len(data21_works)}")
    phase1_experts = data21_long.loc[data21_long["stage"] == "phase1", "expert_code"].nunique()
    if phase1_experts != 42:
        raise ValueError(f"数据2.1第一阶段专家数异常：期望 42，实际 {phase1_experts}")

    stats_df = expert_statistics(data21_long)
    scored_long = add_standardized_scores(data21_long, stats_df)
    calibration_df = calibration_from_data1(data1_long)
    scored_long = attach_weights(scored_long, calibration_df)
    result_df = aggregate_work_scores(scored_long, data21_works)
    elapsed = time.perf_counter() - start
    print("[3/6] 标准化与数据1校准完成", flush=True)

    metrics_df = validation_metrics(result_df, elapsed)
    print("[4/6] 基础验证指标完成", flush=True)
    robustness_df = robustness_metrics(scored_long, data21_works, repeats=10)
    metrics_df = metrics_df.merge(robustness_df[["method", "top50_change_rate_mean", "top50_change_rate_p95"]], on="method")
    print("[5/6] 扰动鲁棒性验证完成", flush=True)

    result_df.to_csv(RESULT_DIR / "q2_work_scores_corrected.csv", index=False, encoding="utf-8-sig")
    scored_long.to_csv(RESULT_DIR / "q2_review_long_corrected.csv", index=False, encoding="utf-8-sig")
    stats_df.merge(calibration_df, on=["stage", "expert_code"], how="left").to_csv(
        RESULT_DIR / "q2_expert_stats_corrected.csv", index=False, encoding="utf-8-sig"
    )
    metrics_df.to_csv(RESULT_DIR / "q2_validation_metrics.csv", index=False, encoding="utf-8-sig")
    robustness_df.to_csv(RESULT_DIR / "q2_robustness_metrics.csv", index=False, encoding="utf-8-sig")

    make_figures(scored_long, result_df, stats_df, metrics_df)
    write_manifest(metrics_df, robustness_df)
    print("[6/6] 图表与复现清单完成", flush=True)

    print(f"作品数：{len(data21_works)}")
    print(f"第一阶段专家数：{phase1_experts}")
    print(f"第二阶段专家数：{data21_long.loc[data21_long['stage'] == 'phase2', 'expert_code'].nunique()}")
    print("\n验证指标：")
    print(metrics_df.to_string(index=False))
    print("\n输出目录：")
    print(f"  {RESULT_DIR}")
    print(f"  {FIG_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
