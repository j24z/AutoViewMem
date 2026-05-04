import json

# 原始数据集文件名
in_file = "locomo10.json"
# 过滤后的新数据集文件名
out_file = "locomo.json"

with open(in_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果最外层是一个列表，列表里的每个元素是 {"qa": [...]}
# 保持原结构，只在 qa 这一层过滤 category == 5 的条目
for block in data:
    if "qa" in block and isinstance(block["qa"], list):
        block["qa"] = [
            item for item in block["qa"]
            if not (isinstance(item, dict) and item.get("category") == 5)
        ]

# 写回新文件，格式和原来一致（只是少了 category=5 的问答）
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("过滤完成，新数据集已保存为:", out_file)
