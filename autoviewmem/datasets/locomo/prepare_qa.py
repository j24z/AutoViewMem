import json
from tqdm import tqdm

# ========================= 配置区 =========================
INPUT_FILE = "locomo.json"          # 输入文件
OUTPUT_FILE = "locomo_qa.json"         # 输出文件（平铺后的纯 QA）
# ===========================================================

def main():
    print(f"正在读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"总共加载了 {len(data)} 条对话数据")

    qa_list = []
    total_qa_before = 0
    total_qa_after = 0

    # 使用 tqdm 显示进度条
    for idx, dialogue in enumerate(tqdm(data, desc="提取 QA", unit="对话")):
        qa_pairs = dialogue.get("qa", [])
        count = len(qa_pairs)
        total_qa_before += count

        if count == 0:
            continue

        # 可选：打印每条对话有多少个问题（调试用，正式运行时可以注释掉）
        # print(f"\n对话 {idx+1:2d} -> 包含 {count} 个 QA")

        for qa in qa_pairs:
            question = str(qa.get("question", "")).strip()
            answer   = str(qa.get("answer", "")).strip()
            category = str(qa.get("category", "")).strip()

            if category == "5":
                continue

            # 过滤空问题或空答案
            if question and answer:
                qa_list.append({
                    "question": question,
                    "answer": answer
                })
                total_qa_after += 1
            else:
                print(question, answer)

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print(f"提取完成！")
    print(f"原始总 QA 数量     : {total_qa_before}")
    print(f"过滤后有效 QA 数量 : {total_qa_after}")
    print(f"已保存到 → {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()