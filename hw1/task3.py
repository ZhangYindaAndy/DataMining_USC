
import sys
import time
import json
from pyspark import SparkContext, StorageLevel



if __name__ == '__main__':
    
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('OFF')

    review_path = sys.argv[1]
    business_path = sys.argv[2]
    output_a_path = sys.argv[3]
    output_b_path = sys.argv[4]
    partition_num = 8

    reviewRDD = sc.textFile(review_path).map(lambda x: json.loads(x))
    reviewRDD = reviewRDD.map(lambda x: (x['business_id'], x['stars']))
    
    businessRDD = sc.textFile(business_path).map(lambda x: json.loads(x))
    businessRDD = businessRDD.map(lambda x: (x['business_id'], x['city']))

    # (business_id, (stars, city)) -> (city, (star, 1))
    dataRDD = reviewRDD.join(businessRDD).map(lambda x: (x[1][1], (x[1][0], 1)))

    # avoid shuffling and reduce partition number 
    # (this will make python sort use less time than spark sort, due to a low partition number)
    # dataRDD = dataRDD.partitionBy(partition_num, lambda x: hash(x) % partition_num)

    # (city, (tot_star, tot_num))
    dataRDD = dataRDD.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    # (city, ave_star)
    # only have around 1000 records (less than 1MB), gather them to one partition 
    dataRDD = dataRDD.map(lambda x: (x[0], float(x[1][0] / x[1][1]))).persist(StorageLevel.DISK_ONLY)
    # trigger the lazy transformation
    dataRDD.take(1)

    # method A
    start = time.time()
    city_list = dataRDD.collect()
    # print(type(city_list), len(city_list))
    city_list = sorted(city_list, key=lambda x: (-x[1], x[0]))

    for each_city in city_list[:10]:
        print(each_city[0], round(each_city[1], 1))
    A_time = time.time() - start

    with open(output_a_path, 'w') as f:
        f.write('city,stars\n')
        for city in city_list:
            f.write('{},{}\n'.format(city[0], city[1]))
  
    # combine to one partition will make spark sort faster
    dataRDD = dataRDD.coalesce(1)
    dataRDD.take(1)

    # method B
    start = time.time()
    city_list = dataRDD.sortBy(lambda x: (-x[1], x[0])).take(10)
    for each_city in city_list:
        print(each_city[0], round(each_city[1], 1))
    B_time = time.time() - start

    print(A_time)
    print(B_time)

    result = {
        'm1': A_time,
        'm2': B_time
    }

    with open(output_b_path, 'w') as f:
        json.dump(result, f)