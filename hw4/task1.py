import os
import sys
import time
import random
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SQLContext
from graphframes import GraphFrame


def read_file(input_path):
    edges = set()
    vertices = set()

    with open(input_path, 'r') as f:
        for line in f.readlines():
            points = line.strip().split(' ')
            edges.add((points[0], points[1]))
            edges.add((points[1], points[0]))  # undirected graph
            vertices.add((points[0], ))  # make it tuple
            vertices.add((points[1], ))

    return list(vertices), list(edges)


if __name__ == '__main__':
 
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.environ['PYSPARK_SUBMIT_ARGS'] = ('--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11')

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')

    start = time.time()

    vertices, edges = read_file(input_path)

    sqlc = SQLContext(sc)
    vertices = sqlc.createDataFrame(vertices, ['id'])
    edges = sqlc.createDataFrame(edges, ['src', 'dst'])
    graph = GraphFrame(vertices, edges) 

    community = graph.labelPropagation(maxIter=5)
    communityRDD = community.rdd.map(lambda x: (x['label'], '\'{}\''.format(x['id']))).groupByKey()
    communityRDD = communityRDD.map(lambda x: sorted(x[1])).sortBy(lambda x: (len(x), x[0]))
    result = communityRDD.collect()

    with open(output_path, 'w') as f:
        for group in result:
            f.write(', '.join(group))
            f.write('\n')

    print('Duration: {}'.format(time.time() - start))


    


