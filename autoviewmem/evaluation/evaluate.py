import argparse
import concurrent.futures
import json
import threading
import re
import time
from collections import defaultdict
from pathlib import Path
import glob
import os
import sys

# 将项目根目录添加到 sys.path 以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoviewmem.evaluation.metrics.llm_judge import evaluate_llm_judge
from autoviewmem.evaluation.metrics.utils import calculate_bleu_scores, calculate_metrics, calculate_em_reward
from autoviewmem.evaluation.metrics.retrieval_eval import calculate_retrieval_metrics
from tqdm import tqdm
import pandas as pd

import nltk


def find_latest_multi_layer_file():
    """查找最新的 routing_*.json 或 multi_layer_routing_*.json 文件"""
    search_patterns = [
        str(Path("results") / "**" / "multi_layer_routing_*.json"),
        str(Path("outputs") / "**" / "multi_layer_routing_*.json"),
        str(Path("results") / "**" / "routing_*.json"),
        str(Path("outputs") / "**" / "routing_*.json"),
    ]
    
    latest_file = None
    latest_time = 0
    
    for pattern in search_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            # 过滤掉 routing_log 文件
            if file_path.endswith("_routing_log.json"):
                continue
                
            file_time = os.path.getmtime(file_path)
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file_path
    
    return latest_file


file_name = "results.json"

# 自动查找最新的 multi_layer_routing_*.json 文件作为默认值
DEFAULT_INPUT_FILE = find_latest_multi_layer_file()
# 如果没有找到最新文件，使用原来的默认值作为 fallback
INPUT_FILE = DEFAULT_INPUT_FILE if DEFAULT_INPUT_FILE else str(Path("results") / file_name)

# 新增：匹配 </think> 及其后空白，用于定位最后一次闭合标签
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
    resp = re.sub(r"^\s*(\*\*\s*)?answer(\s*\*\*)?\s*[:：]\s*(\*\*\s*)?", "", resp, flags=re.IGNORECASE)

    return resp


def process_item(item_data, enable_llm):
    """处理每个 key 的所有 QA 数据"""
    k, v = item_data
    local_results = defaultdict(list)
    processed = 0  # 统计有效处理的条目数

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = clean_response(str(item["response"]))
        category = str(item["category"])
        question = str(item["question"])

        # 跳过类别 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        em_reward = calculate_em_reward(pred_answer, gt_answer, normalize=True)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer) if enable_llm else 0

        # Calculate Retrieval Metrics
        evidence = item.get("evidence", [])
        memories_1 = item.get("speaker_1_memories", [])
        memories_2 = item.get("speaker_2_memories", [])
        
        # Merge and sort memories by score
        all_memories = memories_1 + memories_2
        # Ensure items are dicts and have scores
        all_memories = [m for m in all_memories if isinstance(m, dict)]
        all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        retrieval_metrics = calculate_retrieval_metrics(evidence, all_memories)

        local_results[k].append({
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "em_reward": em_reward,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
            **retrieval_metrics
        })
        processed += 1  # 每写入一条结果即计数

    return local_results, processed


def process_question(task, enable_llm):
    """处理单条 QA 数据"""
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

    # Calculate Retrieval Metrics
    evidence = item.get("evidence", [])
    memories_1 = item.get("speaker_1_memories", [])
    memories_2 = item.get("speaker_2_memories", [])
    
    # Merge and sort memories by score
    all_memories = memories_1 + memories_2
    # Ensure items are dicts and have scores
    all_memories = [m for m in all_memories if isinstance(m, dict)]
    all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    retrieval_metrics = calculate_retrieval_metrics(evidence, all_memories)

    result = {
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        "em_reward": em_reward,
        "bleu_score": bleu_scores["bleu1"],
        "f1_score": metrics["f1"],
        "llm_score": llm_score,
        **retrieval_metrics
    }
    return k, result, 1


def evaluate_and_save(input_file, output_file, max_workers, enable_llm):
    """执行 metrics 计算，并保存 JSON"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    date_str = time.strftime("%Y%m%d")
    
    input_stem = Path(input_file).stem
    
    # Simplify input stem for filename
    # Remove common prefixes
    short_stem = input_stem.replace("multi_layer_routing_", "")
    short_stem = short_stem.replace("multi_layer_", "")
    short_stem = short_stem.replace("routing_", "")
    
    # Remove previous timestamp from input stem if present (usually at end)
    # Regex for _\d{8}_\d{6}$
    short_stem = re.sub(r"_\d{8}_\d{6}$", "", short_stem)
    
    base_name = Path(output_file).stem
    
    new_filename = f"{base_name}_{short_stem}_{timestamp}.json"
    
    # Create daily directory
    output_dir = os.path.join(os.path.dirname(output_file), date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, new_filename)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 扁平化任务列表（按问题并行）
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

    print(f"\n✔ Metrics saved to: {output_file}")
    return output_file


def summarize_results(metrics_file):
    """读取 metrics JSON 并输出统计表"""

    with open(metrics_file, "r") as f:
        data = json.load(f)

    all_items = []
    for key in data:
        all_items.extend(data[key])

    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    metrics_to_agg = {
        "em_reward": "mean", "bleu_score": "mean", "f1_score": "mean", "llm_score": "mean",
        "recall@3": "mean", "recall@5": "mean", "recall@10": "mean",
        "ndcg@3": "mean", "ndcg@5": "mean", "ndcg@10": "mean",
        "redundancy@3": "mean", "redundancy@5": "mean", "redundancy@10": "mean"
    }
    
    # Ensure columns exist
    for col in metrics_to_agg.keys():
        if col not in df.columns:
            df[col] = 0.0

    result = df.groupby("category").agg(metrics_to_agg).round(4)

    result["count"] = df.groupby("category").size()

    print("\n===== Mean Scores Per Category =====")
    print(result)

    print("\n===== Mean Scores Per Conversation =====")
    conv_items = []
    
    cols_conv = ["em_reward", "bleu_score", "f1_score", "llm_score", 
                 "recall@3", "recall@5", "recall@10", "ndcg@3", "ndcg@5", "ndcg@10",
                 "redundancy@3", "redundancy@5", "redundancy@10"]

    for key, items in data.items():
        if not items:
            continue
        sub_df = pd.DataFrame(items)
        # Ensure columns exist in sub_df as well
        for col in cols_conv:
            if col not in sub_df.columns:
                sub_df[col] = 0.0
                
        means = sub_df[cols_conv].mean().round(4)
        means["conversation_id"] = key
        means["count"] = len(items)
        conv_items.append(means)

    if conv_items:
        df_conv = pd.DataFrame(conv_items)
        cols = ["conversation_id", "count"] + cols_conv
        # Convert to string without index
        print(df_conv[cols].to_string(index=False))

    overall_means = df.agg(metrics_to_agg).round(4)

    print("\n===== Overall Mean Scores =====")
    print(overall_means)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results and summarize")
    parser.add_argument("--input_file", default=INPUT_FILE, type=str, help="Input dataset JSON")
    parser.add_argument(
        "--output_file",
        type=str,
        default="results_evaluation/evaluation_metrics.json",
        help="Output metrics JSON",
    )
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument(
        "--enable_llm_judge",
        default=False,
        action="store_true",
        help="开启后调用 LLM 服务器计算 llm_score（受 --max_workers 控制并发）",
    )

    args = parser.parse_args()

    # --- Step 1: Evaluate & Save ---
    metrics_file = evaluate_and_save(
        args.input_file,
        args.output_file,
        args.max_workers,
        args.enable_llm_judge,
    )

    # --- Step 2: Summarize ---
    try:
        summarize_results(metrics_file)
    except Exception as e:
        print(f"\n[Warning] Failed to summarize results: {e}")
        # We don't exit with error here because the main task (generating metrics) succeeded
    
    # 打印测评的原始文件名称
    print(f"\n===== Evaluated File =====")
    print(f"Input file: {args.input_file}")


if __name__ == "__main__":
    main()
