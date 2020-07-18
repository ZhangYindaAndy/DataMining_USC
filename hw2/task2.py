
import sys
import csv
import time
from pyspark import SparkContext, StorageLevel


def preprocess_data(input_path, intermediate_path):
    with open(input_path, 'r') as in_f:
        next(in_f)  # skip the header
        reader = csv.reader(in_f, delimiter=',')
        with open(intermediate_path, 'w') as out_f:
            writer = csv.writer(out_f, delimiter=',')
            for row in reader:
                date_customer_id = '{}-{}'.format(row[0], int(row[1]))
                product_id = int(row[5])
                writer.writerow([date_customer_id, product_id])

def text_parser(x):
    x = x.split(',')
    return (x[0], x[1])


def generate_sizeK_itemsets(itemsets, k):
    # use k-1 candidate itemsets generate unchecked k itemsets
    sizeK_itemsets = []
    for i, set1 in enumerate(itemsets):
        for set2 in itemsets[i+1:]:
            new_set = set(set1) | set(set2)
            if len(new_set) == k:
                sizeK_itemsets.append(tuple(sorted(new_set)))
    return list(set(sizeK_itemsets))


def a_priori(baskets, support, basket_number):
    baskets = list(baskets)
    sub_support = support * len(baskets) / basket_number
    
    # get singleton candidate
    singleton = {}
    for basket in baskets:
        for item in basket:
            if not singleton.get(item, False):
                singleton[item] = 0
            singleton[item] += 1
    singleton = sorted([k for k, v in singleton.items() if v >= sub_support])

    k = 1
    sizeK_itemsets = [tuple([k]) for k in singleton]  # use tuple([k]), or the string will be decomposed
    candidate_itemsets = sizeK_itemsets
    while True:
        k += 1
        sizeK_itemsets = generate_sizeK_itemsets(sizeK_itemsets, k)
        if len(sizeK_itemsets) == 0:
            break
        tmp_sizeK = {}
        for basket in baskets:
            basket_set = set(basket)
            for itemset in sizeK_itemsets:
                if set(itemset).issubset(basket_set):
                    if not tmp_sizeK.get(itemset, False):
                        tmp_sizeK[itemset] = 0
                    tmp_sizeK[itemset] += 1
        sizeK_itemsets = [k for k, v in tmp_sizeK.items() if v >= sub_support]
        candidate_itemsets.extend(sizeK_itemsets)

    yield candidate_itemsets


def count_itemsets(basket, candidate_itemsets):
    itemsets_pair = []
    basket_set = set(basket)
    for itemset in candidate_itemsets:
        if set(itemset).issubset(basket_set):
            itemsets_pair.append((itemset, 1))
    return itemsets_pair


if __name__ == '__main__':
    
    threshold = int(sys.argv[1])  
    support = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    intermediate_path = 'customer_product.csv'

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('OFF')

    # preprocess_data(input_path, intermediate_path)

    start = time.time()
    dataRDD = sc.textFile(intermediate_path).map(lambda x: text_parser(x)).groupByKey().map(lambda x: tuple(set(x[1])))
    basketRDD = dataRDD.filter(lambda x: len(x) > threshold).coalesce(2).persist(StorageLevel.DISK_ONLY)
    basket_number = basketRDD.count()

    # pass 1
    candidate_itemsets = basketRDD.mapPartitions(lambda x: a_priori(x, support, basket_number))
    candidate_itemsets = candidate_itemsets.flatMap(lambda x: x).distinct().sortBy(lambda x: (len(x), x)).collect()
    # print(candidate_itemsets)
    # pass 2
    freq_itemsets = basketRDD.flatMap(lambda x: count_itemsets(x, candidate_itemsets))
    freq_itemsets = freq_itemsets.reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support)
    freq_itemsets = freq_itemsets.map(lambda x: x[0]).sortBy(lambda x: (len(x), x)).collect()

    with open(output_path, 'w') as f:
        f.write('Candidates:\n')
        l = 0
        r = 0
        while r <= len(candidate_itemsets):
            if r == len(candidate_itemsets) or len(candidate_itemsets[r]) > len(candidate_itemsets[l]):
                if len(candidate_itemsets[l]) == 1:
                    str_itemsets = ['(\'{}\')'.format(str(i[0])) for i in candidate_itemsets[l: r]] # remove ending comma
                else:
                    str_itemsets = [str(i) for i in candidate_itemsets[l: r]]
                f.write(','.join(str_itemsets))
                f.write('\n\n')
                l = r
            r += 1 
        f.write('Frequent Itemsets:\n')
        l = 0
        r = 0
        while r <= len(freq_itemsets):
            if r == len(freq_itemsets) or len(freq_itemsets[r]) > len(freq_itemsets[l]):
                if len(freq_itemsets[l]) == 1:
                    str_itemsets = ['(\'{}\')'.format(str(i[0])) for i in freq_itemsets[l: r]]
                else:
                    str_itemsets = [str(i) for i in freq_itemsets[l: r]]
                f.write(','.join(str_itemsets))
                f.write('\n\n')
                l = r              
            r += 1

    print("Duration: {}".format(time.time()-start))

