import argparse 
import concurrent.futures 
import json 
import threading 
import re 
import time 
from collections import defaultdict 
from pathlib import Path 
import os 
import sys 

import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from scipy import stats 

# ===== EMNLP 风格 ===== 
plt.rcParams.update({ 
    "font.family": "serif", 
    "font.serif": ["Times New Roman", "Times"], 
    "font.size": 9, 
    "axes.titlesize": 9, 
    "axes.labelsize": 9, 
    "xtick.labelsize": 8, 
    "ytick.labelsize": 8, 
    "legend.fontsize": 8, 
    "lines.linewidth": 1.0, 
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}) 

# ===== 你的原始 imports 保持不变 ===== 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from autoviewmem.evaluation.metrics.llm_judge import evaluate_llm_judge 
from autoviewmem.evaluation.metrics.utils import calculate_bleu_scores, calculate_metrics, calculate_em_reward 
from autoviewmem.evaluation.metrics.retrieval_eval import calculate_retrieval_metrics 

# Provide input result files with --files. Raw experiment outputs are excluded
# from the public repository.
TARGET_FILES = []

# 匹配 </think> 及其后空白
_THINK_CLOSE_RE = re.compile(r"</think>\s*", flags=re.IGNORECASE)


def clean_response(resp):
    """仅保留最后一个 </think> 之后的内容；若不存在 </think> 则 strip 原文"""
    last = None
    for m in _THINK_CLOSE_RE.finditer(resp):
        last = m

    if last is None:
        resp = resp.strip()
    else:
        resp = resp[last.end():].strip()

    # 过滤 "answer:" / "**Answer:**" 前缀 (不区分大小写)
    resp = resp.lstrip("\ufeff\u200b\u200c\u200d\u2060")
    resp = re.sub(
        r"^\s*(\*\*\s*)?answer(\s*\*\*)?\s*[:：]\s*(\*\*\s*)?",
        "",
        resp,
        flags=re.IGNORECASE
    )
    return resp


def calculate_margin_info(evidence, all_memories):
    """
    计算 margin 和 has_positive
    relevant 判定：memory.source_ids 与 evidence 有交集 => relevant
    s_pos = relevant memories 中的最大 score
    s_neg = non-relevant memories 中的最大 score
    margin = s_pos - s_neg
    """
    evidence_set = set(evidence) if isinstance(evidence, list) else set()

    relevant_scores = []
    non_relevant_scores = []

    for mem in all_memories:
        if not isinstance(mem, dict):
            continue

        mem_source_ids = mem.get("source_ids", [])
        score = mem.get("score", 0.0)

        if not isinstance(mem_source_ids, list):
            mem_source_ids = [mem_source_ids]

        mem_ids_set = set(mem_source_ids)

        if not evidence_set.isdisjoint(mem_ids_set):
            relevant_scores.append(score)
        else:
            non_relevant_scores.append(score)

    has_positive = 1 if relevant_scores else 0

    s_pos = max(relevant_scores) if relevant_scores else None
    s_neg = max(non_relevant_scores) if non_relevant_scores else None

    margin = None
    if s_pos is not None and s_neg is not None:
        margin = float(s_pos - s_neg)

    return margin, has_positive


def process_question(task, enable_llm):
    """处理单条 QA 数据，新增 margin 计算"""
    k, item = task

    gt_answer = str(item["answer"])
    pred_answer = clean_response(str(item["response"]))
    category = str(item["category"])
    question = str(item["question"])

    # 跳过类别 5
    if category == "5":
        return k, None, 0

    metrics = calculate_metrics(pred_answer, gt_answer)
    bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
    em_reward = calculate_em_reward(pred_answer, gt_answer, normalize=True)
    llm_score = evaluate_llm_judge(question, gt_answer, pred_answer) if enable_llm else 0

    # Retrieval metrics
    evidence = item.get("evidence", [])
    memories_1 = item.get("speaker_1_memories", [])
    memories_2 = item.get("speaker_2_memories", [])

    all_memories = memories_1 + memories_2
    all_memories = [m for m in all_memories if isinstance(m, dict)]
    all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)

    retrieval_metrics = calculate_retrieval_metrics(evidence, all_memories)

    # Margin info
    margin, has_positive = calculate_margin_info(evidence, all_memories)

    result = {
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        "em_reward": em_reward,
        "bleu_score": bleu_scores["bleu1"],
        "f1_score": metrics["f1"],
        "llm_score": llm_score,
        "margin": margin,
        "has_positive": has_positive,
        **retrieval_metrics
    }
    return k, result, 1


def bootstrap_ci_mean(arr, n_boot=2000, ci=95, random_state=42):
    """Bootstrap CI for mean."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0]

    rng = np.random.default_rng(random_state)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(means, alpha)
    upper = np.percentile(means, 100 - alpha)
    return float(lower), float(upper)


def bootstrap_ci_rate(arr, n_boot=2000, ci=95, random_state=42):
    """Bootstrap CI for binary rate."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0]

    rng = np.random.default_rng(random_state)
    rates = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        rates.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(rates, alpha)
    upper = np.percentile(rates, 100 - alpha)
    return float(lower), float(upper)


def plot_margin_distribution(df, output_path, title_suffix=""):
    """绘制 Margin 分布图（单文件）"""
    margins = df[df["margin"].notna()]["margin"].astype(float)

    if len(margins) == 0:
        print(f"No valid margin data to plot for {output_path}")
        return

    neg_margin_count = (margins < 0).sum()
    neg_margin_rate = neg_margin_count / len(margins)

    has_pos_mean = df["has_positive"].mean()
    no_pos_rate = 1.0 - has_pos_mean

    mean_val = margins.mean()
    median_val = margins.median()

    plt.figure(figsize=(10, 6))
    plt.hist(margins, bins=30, alpha=0.75, edgecolor="black")

    plt.axvline(0.0, linestyle="--", linewidth=1, label="Zero margin")
    plt.axvline(mean_val, linestyle="-", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    plt.axvline(median_val, linestyle=":", linewidth=1.5, label=f"Median: {median_val:.4f}")

    title_text = (
        f"Margin Distribution {title_suffix}\n"
        f"Mean={mean_val:.4f} | Neg Rate={neg_margin_rate:.2%} | No-Positive Rate={no_pos_rate:.2%}"
    )
    plt.title(title_text)
    plt.xlabel("Margin (s_pos - s_neg)")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔ Margin distribution plot saved to: {output_path}")


def summarize_results(metrics_file):
    """读取 metrics JSON 并输出统计表，新增 margin 和 no_positive_rate"""
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_items = []
    for key in data:
        all_items.extend(data[key])

    if not all_items:
        print("No items to summarize.")
        return

    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    metrics_to_agg = {
        "em_reward": "mean",
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean",
        "recall@3": "mean",
        "recall@5": "mean",
        "recall@10": "mean",
        "ndcg@3": "mean",
        "ndcg@5": "mean",
        "ndcg@10": "mean",
        "margin": "mean",
        "has_positive": "mean"
    }

    for col in metrics_to_agg.keys():
        if col not in df.columns:
            df[col] = np.nan if col == "margin" else 0.0

    result = df.groupby("category").agg(metrics_to_agg).round(4)
    result["count"] = df.groupby("category").size()
    result["no_positive_rate"] = (1 - result["has_positive"]).round(4)
    result = result.rename(columns={"margin": "margin_mean"})

    cols = ["count", "em_reward", "f1_score", "margin_mean", "no_positive_rate", "recall@5", "ndcg@5"]
    cols = [c for c in cols if c in result.columns]

    print("\n===== Mean Scores Per Category (Enhanced) =====")
    print(result[cols])

    overall_means = df.agg(metrics_to_agg).round(4)
    overall_stats = overall_means.to_dict()
    overall_stats["no_positive_rate"] = round(1 - overall_stats.get("has_positive", 0), 4)
    overall_stats.pop("has_positive", None)

    print("\n===== Overall Mean Scores (Enhanced) =====")
    for k, v in overall_stats.items():
        print(f"{k}: {v}")

    output_dir = os.path.dirname(metrics_file)
    base_name = Path(metrics_file).stem
    
    # 保存为 pdf 和 svg
    for ext in ["pdf", "svg"]:
        plot_path = os.path.join(output_dir, f"{base_name}_margin_dist.{ext}")
        plot_margin_distribution(df, plot_path, title_suffix=f"({base_name})")


def evaluate_single_file(input_file, max_workers=128, enable_llm=False):
    """处理单个文件，返回生成的 metrics 文件路径"""
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return None

    print(f"\nProcessing: {input_file}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_path = Path(input_file)
    output_dir = input_path.parent
    output_filename = f"{input_path.stem}_eval_margin_{timestamp}.json"
    output_file = output_dir / output_filename

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = [(k, item) for k, items in data.items() for item in items]
    total_items = sum(1 for _, item in tasks if str(item.get("category", "")) != "5")

    results = defaultdict(list)
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_question, task, enable_llm) for task in tasks]

        with tqdm(total=total_items, desc="Processing QA", unit="item") as pbar:
            for future in concurrent.futures.as_completed(futures):
                k, result, processed = future.result()
                if result is not None:
                    with results_lock:
                        results[k].append(result)
                pbar.update(processed)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✔ Metrics saved to: {output_file}")

    summarize_results(output_file)
    return str(output_file)


# === Helper Functions for Comparison ===

def infer_label_from_filename(filename):
    """根据文件名推断图例 Label (Paper terminology)"""
    base = Path(filename).name
    if "single_view" in base:
        return "Single"
    elif "random" in base:
        return "w/o DPP"
    elif "non_orthogonal" in base:
        return "Non-orth"
    elif "dest" in base:
        return "w/o Cons."
    elif "automem" in base:
        return "Full"
    return "Unknown"


def load_margin_from_metrics(metrics_file):
    """从 metrics json 加载 margin 列表和 has_positive 列表"""
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    margins = []
    has_positive = []

    for _, items in data.items():
        for item in items:
            m = item.get("margin")
            hp = item.get("has_positive", 0)

            if m is not None:
                margins.append(float(m))
            has_positive.append(hp)

    return np.array(margins), np.array(has_positive)


def margin_stats(margins, has_positive_arr):
    """计算统计指标"""
    margins = np.asarray(margins, dtype=float)
    has_positive_arr = np.asarray(has_positive_arr, dtype=float)

    if len(margins) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "neg_margin_rate": 0.0,
            "no_positive_rate": 1.0,
            "mean_ci_low": np.nan,
            "mean_ci_high": np.nan,
            "no_pos_ci_low": np.nan,
            "no_pos_ci_high": np.nan
        }

    neg_count = np.sum(margins < 0)
    neg_rate = neg_count / len(margins)

    pos_rate = np.mean(has_positive_arr) if len(has_positive_arr) > 0 else 0.0
    no_pos_rate = 1.0 - pos_rate

    mean_ci_low, mean_ci_high = bootstrap_ci_mean(margins)
    pos_ci_low, pos_ci_high = bootstrap_ci_rate(has_positive_arr) if len(has_positive_arr) > 0 else (np.nan, np.nan)

    return {
        "n": len(margins),
        "mean": float(np.mean(margins)),
        "median": float(np.median(margins)),
        "neg_margin_rate": float(neg_rate),
        "no_positive_rate": float(no_pos_rate),
        "mean_ci_low": mean_ci_low,
        "mean_ci_high": mean_ci_high,
        "no_pos_ci_low": float(1.0 - pos_ci_high) if not np.isnan(pos_ci_high) else np.nan,
        "no_pos_ci_high": float(1.0 - pos_ci_low) if not np.isnan(pos_ci_low) else np.nan
    }


def cohens_d(x, y):
    """Compute Cohen's d for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    nx = len(x)
    ny = len(y)

    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / pooled_std)


def compare_against_full(margins_by_label):
    """对 Full 和各 ablation 做 Welch t-test + Mann-Whitney U + effect size"""
    full_label = "Full"
    if full_label not in margins_by_label:
        print("Full model not found, skip statistical comparison.")
        return

    full = np.asarray(margins_by_label[full_label], dtype=float)
    if len(full) == 0:
        print("Full model has no margin data, skip statistical comparison.")
        return

    print("\n===== Statistical Tests (vs AutoViewMem Full) =====")
    print(
        f"{'Comparison':35s} "
        f"{'FullMean':>10s} {'OtherMean':>10s} "
        f"{'t_pvalue':>12s} {'mw_pvalue':>12s} {'Cohen_d':>10s}"
    )

    for label, arr in margins_by_label.items():
        if label == full_label:
            continue

        other = np.asarray(arr, dtype=float)
        if len(other) == 0:
            continue

        try:
            _, t_p = stats.ttest_ind(full, other, equal_var=False, nan_policy="omit")
        except Exception:
            t_p = np.nan

        try:
            _, mw_p = stats.mannwhitneyu(full, other, alternative="two-sided")
        except Exception:
            mw_p = np.nan

        d = cohens_d(full, other)

        print(
            f"{label:35s} "
            f"{np.mean(full):10.4f} {np.mean(other):10.4f} "
            f"{t_p:12.4e} {mw_p:12.4e} {d:10.4f}"
        )


# =========================== 
# ✅ 新版：论文级 Boxplot（单栏） 
# =========================== 
def plot_margin_comparison_boxplot(margins_by_label, output_path): 
    preferred_order = [ 
        "Full", 
        "Single", 
        "w/o DPP", 
        "Non-orth", 
        "w/o Cons.", 
    ] 

    labels = [l for l in preferred_order if l in margins_by_label] 
    labels += [l for l in margins_by_label if l not in labels] 
    data = [margins_by_label[l] for l in labels] 

    if not data: 
        return 

    plt.figure(figsize=(3.5, 2.6))  # 单栏尺寸 

    plt.boxplot( 
        data, 
        tick_labels=labels, 
        showmeans=True, 
        patch_artist=False, 
        meanprops=dict(marker="o", markersize=4, markerfacecolor="black", markeredgecolor="black"), 
        medianprops=dict(linewidth=1.2), 
        whiskerprops=dict(linewidth=0.8), 
        capprops=dict(linewidth=0.8), 
        flierprops=dict(marker="o", markersize=2) 
    ) 

    # zero line 
    plt.axhline(0, linestyle="--", linewidth=0.8, color="0.3") 
    
    # y axis limit
    plt.ylim(-0.06, 0.06)

    # annotate mean 
    means = [np.mean(x) for x in data] 
    ns = [len(x) for x in data] 

    for i, (m, n) in enumerate(zip(means, ns), start=1): 
        plt.text(i, m + 0.003, f"{m:+.3f}", ha="center", fontsize=7) 

    plt.ylabel("Margin") 
    plt.xticks(rotation=30, ha="right") 
    plt.title("(a) Margin", fontsize=9, pad=1)

    plt.tight_layout() 
    plt.savefig(output_path, dpi=300, bbox_inches="tight") 
    plt.close() 

    print(f"✔ Saved boxplot: {output_path}") 


# =========================== 
# ✅ 新版：No-positive bar（单栏） 
# =========================== 
def plot_no_positive_rate_bar(stats_by_label, output_path): 
    preferred_order = [ 
        "Full", 
        "Single", 
        "w/o DPP", 
        "Non-orth", 
        "w/o Cons.", 
    ] 

    labels = [l for l in preferred_order if l in stats_by_label] 
    labels += [l for l in stats_by_label if l not in labels] 

    rates = [stats_by_label[l]["no_positive_rate"] for l in labels] 

    plt.figure(figsize=(3.5, 2.6)) 

    x = np.arange(len(labels)) 
    bars = plt.bar(x, rates, edgecolor="black", color="0.6", linewidth=0.8) 

    plt.ylabel("No-positive rate") 
    plt.xticks(x, labels, rotation=30, ha="right") 
    plt.ylim(0.08, 0.20)

    # annotate % 
    for bar, r in zip(bars, rates): 
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height(), 
                 f"{r*100:.1f}%", 
                 ha="center", va="bottom", fontsize=7) 

    plt.title("(b) No-positive rate", fontsize=9, pad=1)

    plt.tight_layout() 
    plt.savefig(output_path, dpi=300, bbox_inches="tight") 
    plt.close() 

    print(f"✔ Saved bar chart: {output_path}") 


# =========================== 
# main（只保留两张图） 
# =========================== 
def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--files", nargs="+") 
    parser.add_argument("--max_workers", type=int, default=32) 
    parser.add_argument("--enable_llm_judge", action="store_true") 
    args = parser.parse_args() 

    files_to_process = args.files if args.files else TARGET_FILES 

    eval_outputs = [] 

    for file_path in files_to_process: 
        out = evaluate_single_file(file_path, args.max_workers, args.enable_llm_judge) 
        if out: 
            eval_outputs.append((file_path, out)) 

    if eval_outputs: 
        margins_by_label = {} 
        stats_by_label = {} 

        for raw_input_path, metrics_path in eval_outputs: 
            label = infer_label_from_filename(raw_input_path) 
            margins, has_pos = load_margin_from_metrics(metrics_path) 
            stats_dict = margin_stats(margins, has_pos) 

            margins_by_label[label] = margins 
            stats_by_label[label] = stats_dict 

        out_dir = str(Path(eval_outputs[0][1]).parent) 

        # 保存为 pdf 和 svg
        for ext in ["pdf", "svg"]:
            plot_margin_comparison_boxplot( 
                margins_by_label, 
                os.path.join(out_dir, f"margin_boxplot_emnlp.{ext}") 
            ) 

            plot_no_positive_rate_bar( 
                stats_by_label, 
                os.path.join(out_dir, f"no_positive_emnlp.{ext}") 
            ) 


if __name__ == "__main__": 
    main()
