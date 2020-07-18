import sys
import time
import random
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def read_file(input_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            tmp_line = []
            # index
            tmp_line.append(int(line[0]))
            # ground truth
            tmp_line.append(int(line[1]))

            for each in line[2:]:
                tmp_line.append(float(each))
            data.append(tmp_line)

    return np.array(data)


def build_label_index_dict(labels):
    label_index_dict = {} # {label: [index_in_chunk]}
    for i, label in enumerate(labels):
        if label not in label_index_dict:
            label_index_dict[label] = []
        label_index_dict[label].append(i)

    return label_index_dict


def build_label_data_dict(labels, chunk_remain):
    label_data_dict = {}
    for i, label in enumerate(labels):
        if label not in label_data_dict:
            label_data_dict[label] = []
        label_data_dict[label].append(chunk_remain[i])

    return label_data_dict


def get_index_in_RS(label_index_dict):
    index_in_RS = []
    for label, indice in label_index_dict.items():
        if len(indice) == 1:
            index_in_RS.append(indice[0])

    return index_in_RS


def generate_statistic_data(label_data_dict):
    # calc N, sum, sumsq for each cluster in DS or CS, and record the label and index
    statistic_data = []  # list of [each_N, each_sum, each_sumsq, index_list, label_list]

    for label, data in label_data_dict.items():
        data = np.array(data)
        each_N = len(data)

        each_sum = np.sum(data[:, 2:], axis=0)
        each_sumsq = np.sum(np.square(data[:, 2:]), axis=0)

        index_list = []
        label_list = []
        for point in data:
            index_list.append(point[0])
            label_list.append(point[1])
        # print(each_sum, each_sumsq)
        # print(each_N, len(index_list), len(label_list))

        statistic_data.append([each_N, each_sum, each_sumsq, index_list, label_list])

    return statistic_data


def generate_CS_data(RS_label_data_dict):
    CS = {}
    for label, data in RS_label_data_dict.items():
        if len(data) > 1:
            CS[label] = data

    return CS


def calc_maha_distance(point, centroid):
    avg = centroid[1] / centroid[0]
    var = (centroid[2] / centroid[0]) - np.square(avg)

    dist = np.square(point - avg) / var
    dist = np.sqrt(np.sum(dist))
    return dist


if __name__ == '__main__':
    
    start = time.time()

    input_path = sys.argv[1]
    n_clusters = int(sys.argv[2])
    output_path = sys.argv[3]

    # step 1: prepare data, load 20% of the data
    data = read_file(input_path)
    num_data = len(data)
    # print(num_data)
    # print(data[0])
    d = data.shape[1] - 2
    threshold = 2 * np.sqrt(d)
    # np.random.shuffle(data)
    chunks = np.array_split(data, 5)
    chunk = chunks[0]

    num_DS_point = []
    num_CS_cluster = []
    num_CS_point = []
    num_RS_point = []

    point_index_in_RS = []

    # step 2: run K-means with 5*n_cluster
    X_chunk = chunk[:, 2:]
    # print(X_chunk.shape)
    model_5k = KMeans(n_clusters=5*n_clusters, random_state=0)
    model_5k.fit(X_chunk)

    # step 3: In the K-means result from step 2, 
    #         move all the clusters that contain only one point to RS
    label_index_dict = build_label_index_dict(model_5k.labels_)
    index_in_RS = get_index_in_RS(label_index_dict)
    RS = chunk[index_in_RS]
    # print('RS: ', RS.shape)
    chunk_remain = np.delete(chunk, index_in_RS, axis=0)
    # print('Chunk remain: ', chunk_remain.shape)

    # step 4: run kmeans again to cluster the rest of the data points with n_cluster
    X_chunk_remain = chunk_remain[:, 2:]
    model_k = KMeans(n_clusters=n_clusters, random_state=0)
    model_k.fit(X_chunk_remain)

    # step 5: use K-means result from step 4 to generate the DS cluster
    #         (discard their points and generate statistics)
    label_data_dict = build_label_data_dict(model_k.labels_, chunk_remain)
    DS = generate_statistic_data(label_data_dict)
    # print(DS)
    # print('DS: ', len(DS))

    # step 6: run K-means on points in RS with 5*n_cluster to generate CS(clusters with more than one points)
    #         and RS(clusters with only one point)
    if len(RS) >= 5*n_clusters:
        # the point in RS is too small, so skip step 6
        model_5k = KMeans(n_clusters=5*n_clusters, random_state=0)
        print(RS[:, 2:].shape)
        model_5k.fit(RS[:, 2:])
        RS_label_index_dict = build_label_index_dict(model_5k.labels_)
        index_in_RS = get_index_in_RS(RS_label_index_dict)
        RS_true = RS[index_in_RS]

        RS_label_data_dict = build_label_data_dict(model_5k.labels_, RS)
        CS_label_data_dict = generate_CS_data(RS_label_data_dict) # {label: [data]}
        CS = generate_statistic_data(CS_label_data_dict)
    else:
        RS_true = RS 
        CS = []

    num_RS_point.append(len(RS_true))

    num_CS_cluster.append(len(CS))
    n_CS_point = 0
    for sub_cs in CS:
        n_CS_point += sub_cs[0]
    num_CS_point.append(n_CS_point)

    n_DS_point = 0
    for sub_ds in DS:
        n_DS_point += sub_ds[0]
        # print(sub_ds[0])
        avg = sub_ds[1] / sub_ds[0]
        var = sub_ds[2] / sub_ds[0] - np.square(avg)
        # print(avg)
        # print(var)
        # print(sub_ds[3])
        # print('label', sub_ds[4][0:20])
    num_DS_point.append(n_DS_point)


    # step 7: load another 20% (4 more round) and repeat 7-12
    for i_chunk in range(1, 5):
        chunk = chunks[i_chunk]

        for point in chunk:
            # print(point)
            point_index = -1
            point_distance = 10 * threshold

            # step 8: for the new pointes, compare them to each of DS using M-distance and assign them
            #        the nearest DS clusters if the distance is < threshold
            for i_DS, centroid in enumerate(DS):
                distance = calc_maha_distance(point[2:], centroid)
                if distance < point_distance:
                    point_distance = distance
                    point_index = i_DS 
            # print(point[1], point_index, point_distance)
            if point_distance < threshold:
                # DS: [each_N, each_sum, each_sumsq, index_list, label_list]
                DS[point_index][0] += 1
                DS[point_index][1] += point[2:]
                DS[point_index][2] += np.square(point[2:])
                DS[point_index][3].append(point[0])
                DS[point_index][4].append(point[1])
            else:
            # step 9: for new points that not assigned to DS clusters, using M-distance and assign 
            #         to neaest CS cluster if the distance is < threshold
                point_cs_index = -1
                point_cs_distance = 10 * threshold
                for i_CS, centriod in enumerate(CS):
                    distance = calc_maha_distance(point[2:], centriod)
                    if distance < point_cs_distance:
                        point_cs_index = i_CS 
                        point_cs_distance = distance
                if point_cs_distance < threshold:
                    CS[point_cs_index][0] += 1
                    CS[point_cs_index][1] += point[2:]
                    CS[point_cs_index][2] += np.square(point[2:])
                    CS[point_cs_index][3].append(point[0])
                    CS[point_cs_index][4].append(point[1])

                else:
                    # step 10: for the new points that are not assigned to a DS or CS, assign to RS
                    RS_true = np.insert(RS_true, 0, values=point, axis=0)
        # print('RS len ', len(RS_true))

        # step 11: run K-means on RS with 5*n_cluster, to generate CS and RS
        if len(RS_true) >= 5*n_clusters:
            model_5k = KMeans(n_clusters=5*n_clusters, random_state=0)
            model_5k.fit(RS_true[:, 2:])
            RS_label_index_dict = build_label_index_dict(model_5k.labels_)
            index_in_RS = get_index_in_RS(RS_label_index_dict)
            tmp_RS_true = RS_true[index_in_RS]

            RS_label_data_dict = build_label_data_dict(model_5k.labels_, RS_true)
            CS_label_data_dict = generate_CS_data(RS_label_data_dict) # {label: [data]}
            tmp_CS = generate_statistic_data(CS_label_data_dict)

            RS_true = tmp_RS_true
        else:
            tmp_CS = []
            # RS_true not change

        # step 12: merge CS that have a M-distance < threshold
        for tmp_cs_centroid in tmp_CS:
            tmp_cs_point = tmp_cs_centroid[1] / tmp_cs_centroid[0]
            merge_index = -1
            merge_distance = 10 * threshold
            for i_CS, cs_centroid in enumerate(CS):
                distance = calc_maha_distance(tmp_cs_point, cs_centroid)
                if distance < merge_distance:
                    merge_distance = distance
                    merge_index = i_CS 
            if merge_distance < threshold:
                CS[merge_index][0] += tmp_cs_centroid[0]
                CS[merge_index][1] += tmp_cs_centroid[1]
                CS[merge_index][2] += tmp_cs_centroid[2]
                CS[merge_index][3].extend(tmp_cs_centroid[3])
                CS[merge_index][4].extend(tmp_cs_centroid[4])
            else:
                # make it as a new CS cluster
                CS.append(tmp_cs_centroid)

        # after last chunk of data, merge CS clusters with DS clusters that distance < threshold
        if i_chunk == 4:
            
            if len(RS_true) > 0:
                point_index_in_RS = list(RS_true[:, 0])

            for cs_centroid in CS:
                cs_point = cs_centroid[1] / cs_centroid[0]
                merge_index = -1
                merge_distance = 10 * threshold
                for i, ds_centroid in enumerate(DS):
                    distance = calc_maha_distance(cs_point, ds_centroid)
                    if distance < merge_distance:
                        merge_distance = distance
                        merge_index = i 
                if merge_distance < threshold:
                    DS[merge_index][0] += cs_centroid[0]
                    DS[merge_index][1] += cs_centroid[1]
                    DS[merge_index][2] += cs_centroid[2]
                    DS[merge_index][3].extend(cs_centroid[3])
                    DS[merge_index][4].extend(cs_centroid[4])
                else:
                    # merge the rest into RS
                    point_index_in_RS.extend(cs_centroid[3])
            # clear the CS
            CS = []

        # calc the output info
        n_DS_point = 0
        for sub_ds in DS:
            # print(sub_ds[0])
            n_DS_point += sub_ds[0]
        num_DS_point.append(n_DS_point)

        num_CS_cluster.append(len(CS))
        n_CS_point = 0
        for sub_cs in CS:
            n_CS_point += sub_cs[0]
        num_CS_point.append(n_CS_point)

        if i_chunk == 4:
            num_RS_point.append(len(point_index_in_RS))
        else:
            num_RS_point.append(len(RS_true))


    print('final RS size ', len(point_index_in_RS))

    # get the result and output
    result_dict = {}
    for label, ds_cluster in enumerate(DS):
        for point_id in ds_cluster[3]:
            result_dict[point_id] = label
    for point_id in point_index_in_RS:
        result_dict[point_id] = -1

    # print(result_dict)
    label_pred = []

    with open(output_path, 'w') as f:
        f.write('The intermediate results:\n')
        for i in range(5):
            f.write('Round {}: {},{},{},{}\n'.format(
                i+1, num_DS_point[i], num_CS_cluster[i], num_CS_point[i], num_RS_point[i]))
        f.write('\n')
        f.write('The clustering results:\n')
        for i in range(num_data):
            label_pred.append(result_dict[i])
            f.write('{},{}\n'.format(i, result_dict[i]))

    origin_DS = 0
    outlier = 0
    origin_cluster = [[] for i in range(d)]
    for point in data:
        if point[1] == -1:
            outlier += 1
        else:
            index = int(point[1])
            origin_cluster[index].append(point[2:])
            origin_DS += 1

    # centroid = []
    # cluster_size = []
    # for i, cluster in enumerate(origin_cluster):
    #     centroid.append([i, np.mean(cluster, axis=0)])
    #     cluster_size.append([i, len(cluster)])

    print('origin outlier num: ', outlier)
    # print('percentage of discard point: ', origin_DS*1.0/num_DS_point[4])

    # print('origin cluster size: ', cluster_size)

    # ds_cluster_size = []
    # for i, cluster in enumerate(DS):
    #     ds_cluster_size.append([int(cluster[4][0]), cluster[0]])
    # print('DS cluster size: ', ds_cluster_size)

    # label_true = data[:, 1].flatten()
    # acc = normalized_mutual_info_score(label_true, label_pred)
    # print('Accuracy: ', acc)


    print('duration: {}'.format(time.time()-start))








