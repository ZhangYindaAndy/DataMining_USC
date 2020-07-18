
import sys
import time
import json
from pyspark import SparkContext, StorageLevel


def count_partition_size(iterator):
    yield sum(1 for item in iterator)


def json_parser(x):
    x = json.loads(x)
    return (x['business_id'], 1)


if __name__ == '__main__':
    
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('OFF')

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # best partition number = 8, 500MB business_id size
    partition_num = int(sys.argv[3])

    jsonRDD = sc.textFile(input_path)
    # dataRDD = jsonRDD.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).persist(StorageLevel.DISK_ONLY)
    dataRDD = jsonRDD.map(json_parser).persist(StorageLevel.DISK_ONLY)

    # default 
    default_part_num = dataRDD.getNumPartitions()
    defalut_part_size = dataRDD.mapPartitions(count_partition_size).collect()

    start = time.time()
    businessRDD = dataRDD.reduceByKey(lambda x, y: x + y).sortBy(lambda x: (-x[1], x[0]))
    top10_business = businessRDD.take(10)
    default_exec_time = time.time() - start

    # customized
    # hash the id to partitions number, will get rid of shuffling across RDD
    # partitionBy must apply on k-v RDD, and partitionFunc will receive the key as input (see source code of partitionBy)
    customizedRDD = dataRDD.partitionBy(partition_num, lambda x: hash(x) % partition_num).persist(StorageLevel.DISK_ONLY)
    customized_part_size = customizedRDD.mapPartitions(count_partition_size).collect()

    start = time.time()
    businessRDD = customizedRDD.reduceByKey(lambda x, y: x + y).sortBy(lambda x: (-x[1], x[0]))
    top10_business = businessRDD.take(10)
    customized_exec_time = time.time() - start

    result = {
        'default': {
            'n_partition': default_part_num,
            'n_items': defalut_part_size,
            'exe_time': default_exec_time
        },
        'customized': {
            'n_partition': partition_num,
            'n_items': customized_part_size,
            'exe_time': customized_exec_time
        }
    }
    with open(output_path, 'w') as f:
        json.dump(result, f)

    print(customized_exec_time)





