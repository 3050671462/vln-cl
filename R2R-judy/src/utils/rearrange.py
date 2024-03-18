import cv2
import json

# 加载 JSON 数据
with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/R2R_train.json', 'r') as f:
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
    
    
    
    paths = data["path"]
    scan = data["scan"]

    all_match_score = 0
    all_num_point = 0

    for i in range(len(paths) - 1):
        image_path1 = "/home/qikang/Matterport3DSimulator/data/v2/" + scan + "/" + paths[i] + "_skybox_small.jpg"
        image_path2 = "/home/qikang/Matterport3DSimulator/data/v2/" + scan + "/" + paths[i + 1] + "_skybox_small.jpg"

        # 计算相邻两个图片的匹配程度
        match_score, num_point = calculate_sift_match(image_path1, image_path2)
        all_match_score += match_score
        all_num_point += num_point

    # 构建新的结果字典
    result_dict = {
        "distance": data["distance"],
        "heading": data["heading"],
        "instructions": data["instructions"],
        "path": data["path"],
        "path_id": data["path_id"],
        "scan": data["scan"],
        "source": "R2R",
        "similarity": all_match_score/len(paths),  # 替换成你的相似度值
        "num_matches": all_num_point,  # 替换成你的匹配数量值
        
    }


    # 将结果写入新的 JSON 文件, 注意匹配点大的放在前面

    if 0 <= all_num_point < 501:
        with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/CLR2Rv4/CLR2R_train_round[5]_v4.json', 'a') as result_file:
            json.dump(result_dict, result_file, indent=2)  # indent参数用于美化输出
            result_file.write(',')
            print("successfully write")
    elif 501 <= all_num_point < 1001:
        with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/CLR2Rv4/CLR2R_train_round[4]_v4.json', 'a') as result_file:
            json.dump(result_dict, result_file, indent=2)  # indent参数用于美化输出
            result_file.write(',')
            print("successfully write")
    elif 1001 <= all_num_point < 1501:
        with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/CLR2Rv4/CLR2R_train_round[3]_v4.json', 'a') as result_file:
            json.dump(result_dict, result_file, indent=2)  # indent参数用于美化输出
            result_file.write(',')
            print("successfully write")
    elif 1501 <= all_num_point < 2001:
        with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/CLR2Rv4/CLR2R_train_round[2]_v4.json', 'a') as result_file:
            json.dump(result_dict, result_file, indent=2)  # indent参数用于美化输出
            result_file.write(',')
            print("successfully write")
    elif 2001 <= all_num_point < 20000:
        with open('/home/qikang/Matterport3DSimulator/tasks/R2R-judy/data/CLR2Rv4/CLR2R_train_round[1]_v4.json', 'a') as result_file:
            json.dump(result_dict, result_file, indent=2)  # indent参数用于美化输出
            result_file.write(',')
            print("successfully write")