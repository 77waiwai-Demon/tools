import pickle
import numpy as np
from tqdm import tqdm


def compute_distance(vec1, vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos = num/denom
    return 1 - (0.5 * cos + 0.5)  # 归一化到[0, 1]区间内


train_path = 'model_feature/company_resnet12_feature_train1.pickle'
test_path = 'model_feature/company_resnet12_feature_test1.pickle'

# train_path = 'model_feature/original_resnet12_feature_train1.pickle'
# test_path = 'model_feature/original_resnet12_feature_test1.pickle'

# train_path = 'model_feature/original_resnet12_arcloss_feature_train1.pickle'
# test_path = 'model_feature/original_resnet12_arcloss_feature_test1.pickle'

# pickle_path = '../top5.pickle'
# pickle_path = '../test_datasets_top5.pickle'
train_dict = pickle.load(open(train_path, 'rb'), encoding='utf-8')
test_dict = pickle.load(open(test_path, 'rb'), encoding='utf-8')

# print(word.keys)
# print(word['/media/ubuntu/lihui/DING/train_temp/080180007/L531421090006_20211031111321126.jpg'][0])
# print(word)
output = {}
# each test_vec compute with each train_vec , get top1 and top5
top1_acc = 0
top5_acc = 0
all_count = 0
test_key_len = len(test_dict.keys())



for test_key in tqdm(test_dict.keys()):
    for test_item in test_dict[test_key]:
        all_count += 1
        top1 = ('class', 999)
        top5 = []

        # compute with each train_vec
        for train_key in train_dict.keys():
            min_dist = 999
            for train_item in train_dict[train_key]:
                dist = compute_distance(test_item, train_item)
                # dist = cosine_similarity(test_item, train_item)
                # update min_dist
                if min_dist > dist:
                    min_dist = dist

            if top1[1] > min_dist:
                top1 = (train_key, min_dist)

            if len(top5) == 0:
                top5_max_index = 0
                top5.append((train_key, min_dist))
            elif 5 > len(top5) > 0:
                top5.append((train_key, min_dist))
                if top5[top5_max_index][1] < min_dist:
                    top5_max_index = len(top5) - 1
            else:
                if top5[top5_max_index][1] > min_dist:
                    top5[top5_max_index] = (train_key, min_dist)
                    for i in range(len(top5)):
                        if top5[top5_max_index][1] < top5[i][1]:
                            top5_max_index = i

        # compute acc
        if test_key == top1[0]:
            top1_acc += 1
        for item in top5:
            if item[0] == test_key:
                top5_acc += 1
                break
print(all_count)
print(top1_acc / all_count)
print(top5_acc / all_count)
