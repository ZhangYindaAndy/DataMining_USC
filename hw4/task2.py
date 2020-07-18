
import sys
import copy
import time
import queue
from itertools import combinations
from pyspark import SparkContext, StorageLevel


def bfs(root, graph):

    bfs_tree = {root: [0, [], 1]}  # {vertex: (level, parent_list, shortest_path_num)}
    # only a single vertex
    if root not in graph:
        return bfs_tree  

    q = queue.Queue()
    q.put(root)

    while not q.empty():
        cur_node = q.get()
        cur_level = bfs_tree[cur_node][0]
        cur_shortest_path_num = bfs_tree[cur_node][2]
        for adj_node in graph[cur_node]:
            if adj_node not in bfs_tree:
                bfs_tree[adj_node] = [cur_level+1, [cur_node], cur_shortest_path_num]
                q.put(adj_node)
            elif bfs_tree[adj_node][0] == cur_level + 1:
                bfs_tree[adj_node][1].append(cur_node)
                bfs_tree[adj_node][2] += cur_shortest_path_num
    return bfs_tree


def calc_between(root, graph, vertrices):

    bfs_tree = bfs(root, graph)
    node_weight = dict([(v, 1) for v in vertrices])
    edge_weight = {}

    level_tree = sorted(bfs_tree.items(), key=lambda x: -x[1][0])
    for node, (level, parent_list, shortest_path_num) in level_tree:
        for parent_node in parent_list:
            edge = (min(node, parent_node), max(node, parent_node))
            parent_path_num = bfs_tree[parent_node][2]
            edge_weight[edge] = node_weight[node] * parent_path_num / float(shortest_path_num)
            node_weight[parent_node] += edge_weight[edge]
            

    return list(edge_weight.items())


def get_communities(graph, vertrices):

    communities = []
    tmp_vertrices = set(vertrices)
    while tmp_vertrices:  # cannot use for loop
        v = list(tmp_vertrices)[0] # get a random node from remaining nodes
        community = set(bfs(v, graph).keys())
        communities.append(community)
        tmp_vertrices = tmp_vertrices - community

    return communities


def calc_modularity(new_graph, origin_graph, degrees, edge_2num):
    
    communities = get_communities(new_graph, vertrices)
    modularity = 0
    # print('communities size: ', len(communities))
    for community in communities:
        for v1, v2 in combinations(community, 2):
            A = 0
            if v2 in origin_graph[v1]:
                A = 1
            k = 1.0 * degrees[v1] * degrees[v2] / edge_2num
            modularity += A - k

    return modularity / edge_2num, communities


def del_edges(new_graph, betweenness):
    
    edges = []
    max_between = betweenness[0][2]
    # print('max betweenness: ', max_between)
    for between in betweenness:
        if between[2] != max_between:
            break
        edges.append((between[0], between[1]))

    for edge in edges:
        new_graph[edge[0]].remove(edge[1])
        new_graph[edge[1]].remove(edge[0])
        if len(new_graph[edge[0]]) == 0:
            del new_graph[edge[0]]
        if len(new_graph[edge[1]]) == 0:
            del new_graph[edge[1]]
    return new_graph


if __name__ == '__main__':
 
    input_path = sys.argv[1]
    between_output_path = sys.argv[2]
    community_output_path = sys.argv[3]

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('OFF')

    start = time.time()

    graphRDD = sc.textFile(input_path).map(lambda x: x.split(' '))
    graphRDD = graphRDD.flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])]).groupByKey().mapValues(lambda x: set(x)).persist(StorageLevel.DISK_ONLY)
    vertricesRDD = graphRDD.map(lambda x: x[0]).repartition(8).distinct().persist(StorageLevel.DISK_ONLY)

    # {node: [adjacent node]}  may change it to set further
    origin_graph = graphRDD.collectAsMap()
    vertrices = vertricesRDD.collect()
    # print(len(orgin_graph))
    # print(vertrices)

    degrees = graphRDD.mapValues(lambda x: len(x)).collectAsMap()
    edge_2num = 0
    for node, degree in degrees.items():
        edge_2num += degree

    # ((v1, v2), betweenness) 
    betweenRDD = vertricesRDD.flatMap(lambda x: calc_between(x, origin_graph, vertrices)).reduceByKey(lambda x, y: x + y)
    betweenRDD = betweenRDD.map(lambda x: (x[0][0], x[0][1], x[1]/2.0)).sortBy(lambda x: (-x[2], x[0], x[1]))
    betweenness = betweenRDD.collect()

    with open(between_output_path, 'w') as f:
        for edge in betweenness:
            f.write('(\'{}\', \'{}\'), {}\n'.format(edge[0], edge[1], edge[2]))

    # find communities

    new_graph = copy.deepcopy(origin_graph)
    opt_communities = []

    max_modularity = -float('inf')
    while len(new_graph) > 0:
        betweenRDD = vertricesRDD.flatMap(lambda x: calc_between(x, new_graph, vertrices)).reduceByKey(lambda x, y: x + y)
        betweenRDD = betweenRDD.map(lambda x: (x[0][0], x[0][1], x[1]/2.0)).sortBy(lambda x: (-x[2], x[0], x[1]))
        betweenness = betweenRDD.collect() 

        modularity, communities = calc_modularity(new_graph, origin_graph, degrees, edge_2num)
        # print('modularity: ', modularity)
        if modularity > max_modularity:
            opt_communities = communities
            max_modularity = modularity
        new_graph = del_edges(new_graph, betweenness)
        # print('new graph size: ', len(new_graph))
        # print('\n')

    for i in range(len(opt_communities)):
        opt_communities[i] = sorted(opt_communities[i])
        for j in range(len(opt_communities[i])):
            opt_communities[i][j] = '\'{}\''.format(opt_communities[i][j])
    opt_communities = sorted(opt_communities, key=lambda x: (len(x), x[0]))

    with open(community_output_path, 'w') as f:
        for group in opt_communities:
            f.write(', '.join(group))
            f.write('\n')

    print('Duration: {}'.format(time.time()-start))



