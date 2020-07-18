import os
import sys
import time
import json
import random
import numpy as np
import xgboost as xgb 

from pyspark import SparkContext, StorageLevel


def text_parser(x):
    x = x.split(',')
    return (x[0], x[1], float(x[2]))

def business_parser(x):
    x = json.loads(x)
    return (x['business_id'], (x['stars'], x['review_count'], x['is_open']))

def user_parser(x):
    x = json.loads(x)
    return (x['user_id'], (x['review_count'], x['average_stars']))

def checkin_parser(x):
    x = json.loads(x)
    times = 0
    for k, v in x['time'].items():
        times += v
    return (x['business_id'], times)

def tip_parser(x):
    x = json.loads(x)
    return ((x['user_id'], x['business_id']), x['likes'])

def photo_parser(x):
    x = json.loads(x)
    return (x['business_id'], 1)

def create_features(x, business_feature, user_feature, checkin_feature, tip_feature, photo_feature, test=False):
    # x: (user, item, rate)
    user = x[0]
    item = x[1]
    feature = []
    feature.extend(business_feature[item])
    feature.extend(user_feature[user])

    if item in checkin_feature:
        feature.append(checkin_feature[item])
    else:
        feature.append(0)
    if (user, item) in tip_feature:
        feature.append(tip_feature[(user, item)])
    else: 
        feature.append(0)
    if item in photo_feature:
        feature.append(photo_feature[item])
    else:
        feature.append(0)

    if not test:
        return ((user, item), (feature, x[2]))
    else:
        return ((user, item), feature)


def round_rate(y):
    if y < 1.0:
        return 1.0
    elif y > 5.0:
        return 5.0
    return y

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
def predict(iterator, item_dict, user_dict, N):
    # cold start
    pearson_memo = {}
    invalid_memo = set()
    result_list = []
    for prediction in iterator:
        user = prediction[0]
        item = prediction[1]

        if user not in user_dict:  # new user
            continue
        elif item not in item_dict:  # new item
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
            continue
        
        pearson_list = [k for k, v in pearson]
        rating_list = [v for k, v in pearson]

        pearson_sum = np.sum(np.abs(pearson_list))
        if pearson_sum == 0:
            continue
 
        # transform [1,5] to [-2,2]
        result = np.dot(pearson_list, np.array(rating_list) - 3) / pearson_sum + 3
        result_list.append(((user, item), (round_rate(result), len(pearson))))
        # print(round_star(result))

    yield result_list


if __name__ == '__main__':
 
    input_folder = sys.argv[1]  
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_2')
    sc.setLogLevel('OFF')

    start = time.time()

    # item-based CF
    trainRDD = sc.textFile(os.path.join(input_folder, 'yelp_train.csv')).filter(lambda x: x != 'user_id,business_id,stars')
    trainRDD = trainRDD.map(text_parser).persist(StorageLevel.DISK_ONLY)

    testRDD = sc.textFile(test_input_path).filter(lambda x: x != 'user_id,business_id,stars')
    testRDD = testRDD.map(text_parser).persist(StorageLevel.DISK_ONLY)

    # {item: {user: rating}}
    item_dict = trainRDD.map(lambda x: (x[1], {x[0]: x[2]})).reduceByKey(lambda x, y: update_dict(x, y)).collectAsMap()
    # {user, [(item, rating)]}
    userRDD = trainRDD.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda x, y: x + y).persist(StorageLevel.DISK_ONLY)
    user_dict = userRDD.collectAsMap()

    N = 10
    testPairRDD = testRDD.repartition(8).mapPartitions(lambda x: predict(x, item_dict, user_dict, N))
    testPairRDD = testPairRDD.flatMap(lambda x: x)
    # {(user, item): (rate, len)}
    item_result = testPairRDD.collectAsMap()

    # model-based

    businessRDD = sc.textFile(os.path.join(input_folder, 'business.json')).map(business_parser)
    business_feature = businessRDD.collectAsMap()
    
    userRDD = sc.textFile(os.path.join(input_folder, 'user.json')).map(user_parser)
    user_feature = userRDD.collectAsMap()
    
    checkinRDD = sc.textFile(os.path.join(input_folder, 'checkin.json')).map(checkin_parser)
    checkin_feature = checkinRDD.collectAsMap()
    
    tipRDD = sc.textFile(os.path.join(input_folder, 'tip.json')).map(tip_parser)
    tip_feature = tipRDD.collectAsMap()

    photoRDD = sc.textFile(os.path.join(input_folder, 'photo.json')).map(photo_parser)
    photoRDD = photoRDD.reduceByKey(lambda x, y: x + y)
    photo_feature = photoRDD.collectAsMap()

    trainRDD = trainRDD.map(lambda x: create_features(x, business_feature, user_feature, checkin_feature, tip_feature, photo_feature))
    train_data = trainRDD.map(lambda x: x[1]).collect()

    print('Load Time: ', time.time()-start)

    x_train = np.array([feature for feature, rate in train_data])
    y_train = np.array([rate for feature, rate in train_data])

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, colsample_bytree=0.8,
                             objective='reg:linear', n_estimators=450, seed=0)
    model.fit(x_train, y_train, eval_metric='rmse')

    testRDD = testRDD.map(lambda x: create_features(x, business_feature, user_feature, checkin_feature, tip_feature, photo_feature))
    test_data = testRDD.collect()

    x_test = []
    y_test = []
    test_key = []
    for (key, (feature, rate)) in test_data:
        test_key.append(key)
        x_test.append(feature)
        y_test.append(rate)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_pred = model.predict(x_test)
    y_pred = [round_rate(y) for y in y_pred]

    # hybrid results
    item_len = []
    for key, val in item_result.items():
        item_len.append(val[1])
    average_len = np.average(item_len)
    max_len = np.max(item_len)
    min_len = np.min(item_len)

    review_trust = []
    for key, val in business_feature.items():
        review_trust.append(val[1])
    average_trust = np.average(review_trust)
    max_trust = np.max(review_trust)
    min_trust = np.min(review_trust)

    n = len(y_pred)
    
    for i in range(n):
        if test_key[i] in item_result:
            weight_item = item_result[test_key[i]][1]
            trust_review = business_feature[test_key[i][1]][1]
            a = 2.0 * (np.log(weight_item) - np.log(min_len)) / (np.log(max_len) - np.log(min_len))
            b = 1.0 * (np.log(trust_review) - np.log(min_trust)) / (np.log(max_trust) - np.log(min_trust))
            # val = a * item_result[test_key[i]][0] + (1 - a) * y_pred[i]
            val = (a / (a + b)) * item_result[test_key[i]][0] + (b / (a + b)) * y_pred[i]
            print((y_pred[i], trust_review), item_result[test_key[i]], val, y_test[i])
            y_pred[i] = val

    print('max trust: ', max_trust)
    print('min trust: ', min_trust)
    print('average trust: ', average_trust)

    print('max len: ', max_len)
    print('ave len: ', average_len)
    print('item_based pair: ', len(item_result))
    RMSE = np.sqrt(np.sum((y_test-y_pred)**2)/n)
    print('RMSE: ', RMSE)

    with open(output_path, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(n):
            f.write('{},{},{}\n'.format(test_key[i][0], test_key[i][1], y_pred[i]))

    print('Duration: {}'.format(time.time()-start))
