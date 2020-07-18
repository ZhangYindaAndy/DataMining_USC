
import sys
import time
import random
import numpy as np
from pyspark import SparkContext, StorageLevel


def text_parser(x):
    x = x.split(',')
    return (x[0], x[1], float(x[2]))


def round_star(star):
    if star < 1.0:
        star = 1.0
    elif star > 5.0:
        star = 5.0
    return star

def update_dict(x, y):
    x.update(y)
    return x

def calc_pearson(r1, r2):

    # co-rated average
    avg1 = np.average(r1)
    avg2 = np.average(r2)

    norm1 = np.linalg.norm(r1-avg1)
    norm2 = np.linalg.norm(r2-avg2)
    if norm1 == 0 or norm2 == 0:
        return 0

    return np.dot(r1-avg1, r2-avg2) / norm1 / norm2


# use redundent calculation to avoid slowshuffling
def predict(iterator, item_dict, user_dict, user_average_rate, item_average_rate, all_average_rate, N):
    # cold start
    pearson_memo = {}
    invalid_memo = set()
    result_list = []
    for prediction in iterator:
        user = prediction[0]
        item = prediction[1]

        if user not in user_dict and item not in item_dict:  # new user
            result_list.append((user, item, all_average_rate))
            continue
        elif user not in user_dict:
            result_list.append((user, item, item_average_rate[item]))
            continue
        elif item not in item_dict:  # new item
            result_list.append((user, item, user_average_rate[user]))
            continue

        user_item_list = user_dict[user]
        pearson = []
        for relate_item, rating in user_item_list:
            if (item, relate_item) in invalid_memo or (relate_item, item) in invalid_memo:
                continue
            if (item, relate_item) in pearson_memo:
                pearson.append((pearson_memo[(item, relate_item)], rating))
                continue
            if (relate_item, item) in pearson_memo:
                pearson.append((pearson_memo[(relate_item, item)], rating))
                continue

            i1_userdict = item_dict[item]
            i2_userdict = item_dict[relate_item]
            co_user = set(i1_userdict.keys()) & set(i2_userdict.keys())

            # can use user total average 
            rate_item1 = []
            rate_item2 = []
            for u in co_user:
                rate_item1.append(i1_userdict[u])
                rate_item2.append(i2_userdict[u])

            # support of the item pair
            if len(rate_item1) <= 8:
                invalid_memo.add((item, relate_item))
                invalid_memo.add((relate_item, item))
                continue 

            weight = calc_pearson(rate_item1, rate_item2)
            # remove unrelated (don't need)
            if abs(weight) < 0.25:
                invalid_memo.add((item, relate_item))
                invalid_memo.add((relate_item, item))
                continue
            pearson.append((weight, rating))
            pearson_memo[(relate_item, item)] = weight
            pearson_memo[(item, relate_item)] = weight

        # try to use all items
        if len(pearson) <= N:  # no related items
            result_list.append((user, item, item_average_rate[item])) 
            continue
        # pearson = sorted(pearson, key=lambda x: -x[0])[:N]
        
        pearson_list = [k for k, v in pearson]
        rating_list = [v for k, v in pearson]

        pearson_sum = np.sum(np.abs(pearson_list))
        if pearson_sum == 0:
            result_list.append((user, item, item_average_rate[item])) 
            continue
 
        # transform [1,5] to [-2,2]
        result = np.dot(pearson_list, np.array(rating_list) - 3) / pearson_sum + 3
        result_list.append((user, item, round_star(result)))
        # print(round_star(result))

    yield result_list


if __name__ == '__main__':
 
    train_input_path = sys.argv[1]  
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_1')
    sc.setLogLevel('OFF')

    start = time.time()

    # (user, item, rate)
    trainRDD = sc.textFile(train_input_path).filter(lambda x: x != 'user_id,business_id,stars')
    trainRDD = trainRDD.map(lambda x: text_parser(x)).persist(StorageLevel.DISK_ONLY) # ok here

    testRDD = sc.textFile(test_input_path).filter(lambda x: x != 'user_id,business_id,stars')
    testRDD = testRDD.map(lambda x: text_parser(x)).persist(StorageLevel.DISK_ONLY)
    # only calc the items show in test

    # {item: {user: rating}}
    # item_dict = trainRDD.map(lambda x: (x[1], {x[0]: x[2]})).reduceByKey(lambda x, y: update_dict(x, y)).collectAsMap()
    itemRDD = trainRDD.map(lambda x: (x[1], {x[0]: x[2]})).reduceByKey(lambda x, y: update_dict(x, y)).persist(StorageLevel.DISK_ONLY)
    item_dict = itemRDD.collectAsMap()
    # {user, [(item, rating)]}
    userRDD = trainRDD.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda x, y: x + y).persist(StorageLevel.DISK_ONLY)
    user_dict = userRDD.collectAsMap()

    # cold start: calc user average rating for new item, all average for new user
    user_average_rate = userRDD.mapValues(lambda x: np.average([r[1] for r in x])).collectAsMap()   # 0.5s
    item_average_rate = itemRDD.mapValues(lambda x: np.average([v for k, v in x.items()])).collectAsMap()
    all_average_rate = trainRDD.map(lambda x: x[2]).mean() # 3s
  
    N = 10
    # testPairRDD = testRDD.repartition(16).map(lambda x: predict(x[0], x[1], item_dict, user_dict, user_average_rate, all_average_rate, N)).persist(StorageLevel.DISK_ONLY)
    testPairRDD = testRDD.repartition(8).mapPartitions(lambda x: predict(x, item_dict, user_dict, user_average_rate, item_average_rate, all_average_rate, N))
    testPairRDD = testPairRDD.flatMap(lambda x: x).persist(StorageLevel.DISK_ONLY)

    result = testPairRDD.collect()
    with open(output_path, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for pair in result:
            f.write('{},{},{}\n'.format(pair[0], pair[1], pair[2]))

    print('Duration: {}'.format(time.time() - start))

    testRDD = testRDD.map(lambda x: ((x[0], x[1]), x[2]))
    RMSE = testPairRDD.map(lambda x: ((x[0], x[1]), x[2])).join(testRDD).map(lambda x: x[1][0]-x[1][1]).collect()
    n = len(RMSE)  # 142044
    print('RMSE len: ', n)

    RMSE = np.sqrt(np.sum(np.array(RMSE) ** 2) / n)
    print('RMSE: {}'.format(RMSE))



