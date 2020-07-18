
import sys
import time
import random
from pyspark import SparkContext, StorageLevel


prime = [353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 
         487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 
         631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 
         773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 
         937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 
         1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 
         1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 
         1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 
         1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 
         1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 
         1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 
         1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 
         2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137,
         2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 
         2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 
         2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579]


def text_parser(x):
    x = x.split(',')
    return (x[1], x[0]) 


def create_feature(x, user_mapping):
    user_set = []
    for user in x:
        user_set.append(user_mapping[user])
    return tuple(user_set)


def generate_minhash_func(minhash_num):
    # f(x) = ((ax + b) % p) % m
    # p is any prime number, m is the number of bins
    random.seed(0)
    a = random.sample(range(1, 10000), minhash_num)
    b = random.sample(range(1, 10000), minhash_num)
    p = random.sample(prime, minhash_num)

    return (a, b, p)


def create_minhash(user_set, bins_num, minhash_num, minhash_func):

    calc_hash = lambda a, x, b, p: (a * x + b) % p

    minhash = [2 * minhash_num for i in range(minhash_num)]
    for user_id in user_set:
        for hash_id in range(minhash_num):
            hash_val = calc_hash(minhash_func[0][hash_id], user_id, minhash_func[1][hash_id], minhash_func[2][hash_id]) % minhash_num
            if hash_val < minhash[hash_id]:
                minhash[hash_id] = hash_val
                    
    return tuple(minhash)


def create_bands(business_id, minhash, bands, rows):
    bands_data = []
    for b in range(bands):
        bands_data.append((b, (business_id, minhash[b*rows:b*rows+rows])))
    return bands_data


def create_buckets(band_list, bucket_num):
    bucket_list = [[] for i in range(bucket_num)]
    for business_id, minhash_band in band_list:
        bucket_id = hash(tuple(minhash_band)) % bucket_num
        bucket_list[bucket_id].append(business_id)
    return bucket_list


def generate_pairs(bucket):
    # each pair in alphabetic order
    bucket = sorted(bucket)  
    pair_list = []
    for i, b1 in enumerate(bucket):
        for b2 in bucket[i+1:]:
            pair_list.append((b1, b2))
    return pair_list


def calc_jaccard(pair, features):
    f1 = set(features[pair[0]])
    f2 = set(features[pair[1]])
    return (pair, len(f1 & f2) * 1.0 / len(f1 | f2))


if __name__ == '__main__':
 
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')

    start = time.time()

    dataRDD = sc.textFile(input_path).filter(lambda x: x != 'user_id,business_id,stars')
    # (business_id, user_id)
    dataRDD = dataRDD.map(lambda x: text_parser(x)).persist(StorageLevel.DISK_ONLY)

    user_list = dataRDD.map(lambda x: x[1]).distinct().collect()
    bins_num = len(user_list)
    # print(bins_num) # 11270 num of users
    user_mapping = {}
    for i, user in enumerate(user_list):
        user_mapping[user] = i

    businessRDD = dataRDD.groupByKey().mapValues(lambda x: create_feature(x, user_mapping)).persist(StorageLevel.DISK_ONLY)
    # {business_id: list(user_id)}
    features = dict(businessRDD.collect())
    # print(len(features)) # 24732 num of business

    # can reduce the bands num to reduce time
    bands = 64
    rows = 4
    minhash_num = bands * rows
    minhash_func = generate_minhash_func(minhash_num)

    bucket_num = 40000
    # (business_id, minhash)
    minhashRDD = businessRDD.repartition(32).mapValues(lambda x: create_minhash(x, bins_num, minhash_num, minhash_func))
    # (band_id, list(business_id, minhash_band)) for LSH
    bandRDD = minhashRDD.flatMap(lambda x: create_bands(x[0], x[1], bands, rows)).groupByKey()
    # all buckets of each band, select buckets contain more than 2 pairs
    bucketRDD = bandRDD.flatMap(lambda x: create_buckets(x[1], bucket_num)).filter(lambda x: len(x) >= 2)
    # all pairs generated from buckets
    pairRDD = bucketRDD.flatMap(lambda x: generate_pairs(x)).distinct()
    # result in alphabetic order
    pairRDD = pairRDD.map(lambda x: calc_jaccard(x, features)).filter(lambda x: x[1] >= 0.5).sortBy(lambda x: (x[0][0], x[0][1])).persist(StorageLevel.DISK_ONLY)
    pairs = pairRDD.collect()

    with open(output_path, 'w') as f:
        f.write('business_id_1, business_id_2, similarity\n')
        for pair in pairs:
            f.write('{},{},{}\n'.format(pair[0][0], pair[0][1], pair[1]))

    print('Duration: {}'.format(time.time() - start))

    result_pair = pairRDD.map(lambda x: x[0])
    truth_pair = sc.textFile('pure_jaccard_similarity.csv').filter(lambda x: x != 'business_id_1, business_id_2, similarity')
    truth_pair = truth_pair.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1])).persist(StorageLevel.DISK_ONLY)

    tp = truth_pair.intersection(result_pair).count()
    fp = result_pair.subtract(truth_pair).count()
    fn = truth_pair.subtract(result_pair).count()

    print('precision: {}'.format(tp*1.0/(tp+fp)))
    print('recall: {}'.format(tp*1.0/(tp+fn)))