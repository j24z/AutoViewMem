import argparse
import json
import glob
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
from datetime import datetime

def parse_timestamp(filename):
    # Try to extract timestamp YYYYMMDD_HHMMSS
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return "Unknown"

def get_file_stats(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    memory_ratios = defaultdict(list)
    overall_ratios = []
    
    total_items = 0
    
    # Check if data is dict (grouped by id) or list
    items_list = []
    if isinstance(data, dict):
        for key, items in data.items():
            if isinstance(items, list):
                items_list.extend(items)
            else:
                # Handle case where items might not be a list
                pass
    elif isinstance(data, list):
        items_list = data
    
    for item in items_list:
        category = str(item.get("category", "Unknown"))
        if category == "5": 
            continue
            
        s1_mems = item.get("speaker_1_memories", [])
        s2_mems = item.get("speaker_2_memories", [])
        
        # Ensure s1_mems/s2_mems are lists
        if not isinstance(s1_mems, list): s1_mems = []
        if not isinstance(s2_mems, list): s2_mems = []

        all_mems = s1_mems + s2_mems
        
        counts = {"L0": 0, "L1": 0, "L2": 0}
        for mem in all_mems:
            if isinstance(mem, dict):
                layer = mem.get("layer")
                if layer in counts:
                    counts[layer] += 1
        
        total_count = sum(counts.values())
        if total_count > 0:
            ratios = [counts["L0"]/total_count, counts["L1"]/total_count, counts["L2"]/total_count]
            memory_ratios[category].append(ratios)
            overall_ratios.append(ratios)
        
        total_items += 1
            
    # Calculate means
    stats = {
        "filename": os.path.basename(filepath),
        "timestamp": parse_timestamp(filepath),
        "total_items": total_items,
        "overall": np.mean(overall_ratios, axis=0) if overall_ratios else [0, 0, 0],
        "categories": {}
    }
    
    for cat, ratios in memory_ratios.items():
        stats["categories"][cat] = np.mean(ratios, axis=0)
        
    return stats

def analyze_all_files():
    parser = argparse.ArgumentParser(description="Compare layer distribution across multiple result files")
    parser.add_argument("--input_dir", type=str, default="outputs", help="Directory to search")
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files. Analyzing...")
    
    all_stats = []
    for f in files:
        stats = get_file_stats(f)
        if stats:
            all_stats.append(stats)
            
    # Sort by timestamp
    all_stats.sort(key=lambda x: x["timestamp"])
    
    # 1. Overall Comparison Table
    print("\n" + "="*100)
    print(f"OVERALL DISTRIBUTION COMPARISON ({len(all_stats)} files)")
    print("="*100)
    print(f"{'Timestamp':<16} | {'L0':<7} | {'L1':<7} | {'L2':<7} | {'Items':<5} | {'File (Short)'}")
    print("-" * 100)
    
    for s in all_stats:
        r = s["overall"]
        fname = s["filename"]
        # Shorten filename for display
        short_name = fname.replace("multi_layer_routing_", "")
        if len(short_name) > 40:
            short_name = "..." + short_name[-37:]
            
        print(f"{s['timestamp']:<16} | {r[0]:.1%}   | {r[1]:.1%}   | {r[2]:.1%}   | {s['total_items']:<5} | {short_name}")

    # 2. Per-Category Comparison Tables
    # Collect all categories present
    all_cats = set()
    for s in all_stats:
        all_cats.update(s["categories"].keys())
    
    sorted_cats = sorted(list(all_cats), key=lambda x: int(x) if x.isdigit() else 999)
    
    for cat in sorted_cats:
        print("\n" + "="*60)
        print(f"CATEGORY {cat} DISTRIBUTION")
        print("="*60)
        print(f"{'Timestamp':<16} | {'L0':<7} | {'L1':<7} | {'L2':<7}")
        print("-" * 60)
        
        for s in all_stats:
            if cat in s["categories"]:
                r = s["categories"][cat]
                print(f"{s['timestamp']:<16} | {r[0]:.1%}   | {r[1]:.1%}   | {r[2]:.1%}")
            else:
                # If category is missing in this file, skip or print N/A
                # print(f"{s['timestamp']:<16} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7}")
                pass

if __name__ == "__main__":
    analyze_all_files()
