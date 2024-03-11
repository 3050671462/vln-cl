import json

# 加载 JSON 数据
with open('your_dataset.json', 'r') as f:
    dataset = json.load(f)

def calculate_similarity(kp1, kp2, matches):
    return len(matches) / min(len(kp1), len(kp2))

#函数 calculate_sift_match，接受两个图像路径并返回它们的SIFT匹配程度
def calculate_sift_match(image_path1, image_path2):
    # 读取图像
    im1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher默认参数
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 计算相似度
    similarity = calculate_similarity(kp1, kp2, good)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(im1, kp1, im2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配结果和相似度
    print(f"Similarity: {similarity:.2%}")
    num_matches = len(good)
    print(f"Number of matches: {num_matches}")

    return similarity, num_matches





# 遍历数据集，计算相邻两个图片的匹配程度
for data in dataset:
    
    # 用于存储结果的列表
    results = []
    
    paths = data["path"]
    
    all_match_score = 0
    all_num_point = 0

    for i in range(len(paths) - 1):
        image_path1 = "data/..." + paths[i]
        image_path2 = "data/..." + paths[i + 1]

        # 计算相邻两个图片的匹配程度
        match_score, num_point = calculate_sift_match(image_path1, image_path2)
        all_match_score += match_score
        all_num_point += num_point

    # 构建新的结果字典
    result_dict = {
        "similarity": all_match_score/len(paths),  # 替换成你的相似度值
        "num_matches": all_num_point,  # 替换成你的匹配数量值
        "distance": data["distance"],
        "scan": data["scan"],
        "path_id": data["path_id"],
        "path": data["paths"]
    }

        # 添加到结果列表
    results.append(result_dict)

    # 将结果写入新的 JSON 文件
    with open('CLR2R_train_SIFT.json', 'w') as result_file:
        json.dump(results, result_file, indent=2)  # indent参数用于美化输出



