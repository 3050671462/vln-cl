import json

# 加载 JSON 数据
with open('E:\PAPER\\vln-cl\R2R-judy\data\R2R_train.json', 'r') as f:
    dataset = json.load(f)

def count_dir(instruction):
    # 定义方向词汇表
    direction_words = ["forward", "down", "left", "right", "straight", "past", "towards"]

    # 统计方向词频
    direction_num = sum(instruction.count(word) for word in direction_words)

    # 结果
    return direction_num


results = []

for data in dataset:
    
    
    instruction = data["instructions"]

    direction_num = 0

    for i in range(len(instruction) ):
        direction_num += count_dir(instruction[i])
        
    print(f"总方向变化次数为：{direction_num}")


        # 构建新的结果字典
    result_dict = {
        "direction_num": direction_num,
        "distance": data["distance"],
        "scan": data["scan"],
        "path_id": data["path_id"]
    }

        # 添加到结果列表
    results.append(result_dict)

    # 将结果写入新的 JSON 文件
with open('CLR2R_train_direction_num.json', 'w') as result_file:
    json.dump(results, result_file, indent=2)  # indent参数用于美化输出
