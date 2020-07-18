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

    return ((user, item), (feature, x[2]))
    # if not test:
    #     return ((user, item), (feature, x[2]))
    # else:
    #     return ((user, item), feature)


def round_rate(y):
    if y < 1.0:
        return 1.0
    elif y > 5.0:
        return 5.0
    return y


if __name__ == '__main__':
 
    input_folder = sys.argv[1]  
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    sc = SparkContext('local[*]', 'task2_2')
    sc.setLogLevel('OFF')

    start = time.time()

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

    trainRDD = sc.textFile(os.path.join(input_folder, 'yelp_train.csv')).filter(lambda x: x != 'user_id,business_id,stars')
    trainRDD = trainRDD.map(text_parser)
    trainRDD = trainRDD.map(lambda x: create_features(x, business_feature, user_feature, checkin_feature, tip_feature, photo_feature))
    train_data = trainRDD.map(lambda x: x[1]).collect()

    print('Load Time: ', time.time()-start)

    x_train = np.array([feature for feature, rate in train_data])
    y_train = np.array([rate for feature, rate in train_data])

    # print(x_train.shape, y_train.shape)

    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # param = {'max_depth': 10, 'eta': 0.1, 'colsample_bytree': 0.8, 'eval_metric': 'rmse'}
    # model = xgb.train(param, dtrain, num_boost_round=100)
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, colsample_bytree=0.8,
                             objective='reg:linear', n_estimators=450, seed=0)
    model.fit(x_train, y_train, eval_metric='rmse')

    testRDD = sc.textFile(test_input_path).filter(lambda x: x != 'user_id,business_id,stars')
    testRDD = testRDD.map(text_parser)
    testRDD = testRDD.map(lambda x: create_features(x, business_feature, user_feature, checkin_feature, tip_feature, photo_feature))
    test_data = testRDD.collect()

    x_test = []
    y_test = []
    y_key = []
    for (key, (feature, rate)) in test_data:
        y_key.append(key)
        x_test.append(feature)
        y_test.append(rate)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # dtest = xgb.DMatrix(x_test)
    # y_pred = model.predict(dtest)
    y_pred = model.predict(x_test)
    y_pred = [round_rate(y) for y in y_pred]

    n = len(y_pred)
    RMSE = np.sqrt(np.sum((y_test-y_pred)**2)/n)
    print('RMSE: ', RMSE)
    # print(xgb.importance(model))

    with open(output_path, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(n):
            f.write('{},{},{}\n'.format(y_key[i][0], y_key[i][1], y_pred[i]))

    print('Duration: {}'.format(time.time()-start))
