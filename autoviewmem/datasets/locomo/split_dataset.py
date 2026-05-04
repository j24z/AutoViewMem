import json

# 原始（已过滤）数据集
in_file = "locomo.json"

# 输出三个文件
train_file = "locomo_train.json"
val_file   = "locomo_val.json"
test_file  = "locomo_test.json"

with open(in_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 默认：第一个对话 -> 训练集
#      第二个对话 -> 验证集
#      其余对话   -> 测试集
train_data = []
val_data = []
test_data = []

if len(data) >= 1:
    train_data = [data[0]]
if len(data) >= 2:
    val_data = [data[1]]
if len(data) > 2:
    test_data = data[2:]
else:
    test_data = []

# 写出三个文件
with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(val_file, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("完成划分：")
print("  训练集:", train_file)
print("  验证集:", val_file)
print("  测试集:", test_file)
