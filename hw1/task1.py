
import sys
import json
import time
from pyspark import SparkContext, StorageLevel


def json_parser(x):
    x = json.loads(x)
    return (x['user_id'], x['business_id'], x['date'])


if __name__ == '__main__':
    
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # review.json is 5GB, user_id only consist 500MB, each partition should be 64MB
    partition_num = 8
    start = time.time()

    jsonRDD = sc.textFile(input_path)
    # json.loads will transfer json-like string to dict
    # dataRDD = jsonRDD.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'], x['date']))
    dataRDD = jsonRDD.map(json_parser)  # process together is faster, don't need to save the middle result
    dataRDD = dataRDD.coalesce(2 * partition_num).persist(StorageLevel.DISK_ONLY)
    n_review = dataRDD.count()
    n_review_2018 = dataRDD.filter(lambda x: x[2][:4] == '2018').count()


    userRDD = dataRDD.map(lambda x: (x[0], 1)).partitionBy(partition_num, lambda x: hash(x) % partition_num)
    userRDD = userRDD.reduceByKey(lambda x, y: x + y).sortBy(lambda x: (-x[1], x[0])).persist(StorageLevel.DISK_ONLY)
    n_user = userRDD.count()
    top10_user = userRDD.take(10)
    userRDD.unpersist()

    businessRDD = dataRDD.map(lambda x: (x[1], 1)).partitionBy(partition_num, lambda x: hash(x) % partition_num)
    businessRDD = businessRDD.reduceByKey(lambda x, y: x + y).sortBy(lambda x: (-x[1], x[0])).persist(StorageLevel.DISK_ONLY)
    n_business = businessRDD.count()
    top10_business = businessRDD.take(10)
    businessRDD.unpersist()

    result = {
        'n_review': n_review,
        'n_review_2018': n_review_2018,
        'n_user': n_user,
        'top10_user': [list(user) for user in top10_user],
        'n_business': n_business,
        'top10_business': [list(business) for business in top10_business]
    }
    with open(output_path, 'w') as f:
        json.dump(result, f)

    print(time.time() - start)

