from __future__ import division
import os, sys, math
import matplotlib.pyplot as plt
import numpy as np
import decimal
decimal.setcontext(decimal.Context(prec=34))


class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return ",".join(map(str, self.decisions)) + "|" + str(self.objective)


def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance


def similarity(list1, list2):
    """higher score indicates they are very different"""
    same = 0
    diff = 0
    for a, b in zip(list1, list2):
        if a != b: diff += 1
        else: same += 1
    number1 = int("".join(map(str, list1)))
    number2 = int("".join(map(str, list2)))
    return math.exp(diff)/(math.exp(same) * (number1 - number2))

def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]


def generate_hist(data):
    members = sorted(set(data), reverse=True)
    d = {}
    for member in members:
        d[str(member)] = [i for i, dd in enumerate(data) if dd == member]
    return d


def get_indices(data, count_members):
    indices_dict = generate_hist(data)
    return_indices = []
    for key in indices_dict.keys():
        length_next = len(indices_dict[key])
        if len(return_indices) + length_next <= count_members: return_indices.extend(indices_dict[key])
        else:
            delta = count_members - len(return_indices)
            assert(delta < len(indices_dict[key])), "somethign is wrong"
            from random import shuffle
            shuffle(indices_dict[key])
            delta_e = indices_dict[key][:delta]
            return_indices.extend(delta_e)
            break
    assert(len(return_indices) == count_members), "Somethign is wrong"

    return return_indices


def find_hotspot(clusters, count_members):

    each_cluster_count = int(count_members/len(clusters))

    indexes = []
    for c in xrange(len(clusters)):
        sorted_list = sorted(clusters[c], key=lambda x: x.hotspot_score, reverse=True)
        indexes.extend([s.id for s in sorted_list[:each_cluster_count]])

    assert(len(indexes) == count_members), "somethign is wrong"
    return indexes


def read_csv(filename, header=False):
    def transform(filename):
        return "./Data/input/" + filename

    import csv
    data = []
    f = open(transform(filename), 'rb')
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if i == 0 and header is False: continue  # Header
        elif i ==0 and header is True:
            H = row
            continue
        data.append(data_item(i-1, [1 if x == "Y" else 0 for x in row[:-1]], float(row[-1]) * (10**4))) # TODO: DecisionTree regressor returns int values. As a work around I multiply all the class values by 10**4
    f.close()
    if header is True: return H, data
    return data


def get_clusters(data, no_of_clusters=4):
    from sklearn.cluster import KMeans
    decision_data = [d.decisions for d in data]
    K = KMeans(n_clusters=no_of_clusters)
    y_pred = K.fit_predict(decision_data)

    clusters = [[] for _ in xrange(no_of_clusters)]
    for d, cid in zip(data, y_pred):
        d.cluster_id = cid
        clusters[d.cluster_id].append(d)

    return clusters


def add_hotspot_scores(data, no_of_clusters):
    clusters = get_clusters(data, no_of_clusters)

    distance_matrix = [[-1 for _ in xrange(len(data))] for _ in xrange(len(data))]
    for cluster_no, cluster in enumerate(clusters):
        from sklearn.metrics import jaccard_similarity_score
        for i in xrange(len(cluster)):
            for j in xrange(len(cluster)):
                if distance_matrix[i][j] == -1 and i != j:
                    distance_matrix[i][j] = decimal.Decimal(jaccard_similarity_score(cluster[i].decisions, cluster[j].decisions))
                    distance_matrix[j][i] = distance_matrix[i][j]
                elif distance_matrix[i][j] == -1 and i == j:
                    distance_matrix[j][i] = 1
                else:
                    pass
        for i in xrange(len(cluster)):
            clusters[cluster_no][i].hotspot_score = sum(distance_matrix[i])
        print "Done calculating hotspot scores"
    return clusters


def run_experiment(dataset_name, cluster_numbers=None):
    from time import time
    from os import system
    system("mkdir ./Data/output_knn_hotspot/" + dataset_name)
    system("mkdir ./Data/output_knn_hotspot/" + dataset_name + "/data/")
    output_file = "./Data/output_knn_hotspot/" + dataset_name + "/data/" + dataset_name + "_" + str(cluster_numbers)
    data = read_csv(dataset_name)
    clusters = add_hotspot_scores(data, cluster_numbers)

    output_string = ""
    for count_member in range(cluster_numbers,  int(len(data)* 0.2))[::cluster_numbers]: # ensuring every cluster has same number of representatives
        print ". ",
        sys.stdout.flush()
        training_indices =  find_hotspot(clusters, count_member)
        print len(training_indices), len(data)
        try:
            training_set = [data[i] for i in training_indices]
        except:
            import pdb
            pdb.set_trace()
        indep_training_set = [td.decisions for td in training_set]
        dep_training_set = [td.objective for td in training_set]

        testing_indices = diff(range(len(data)), training_indices)
        testing_set = [data[i] for i in testing_indices]
        assert(len(training_indices) + len(testing_indices) == len(data)), "something is wrong"
        indep_testing_set = [td.decisions for td in testing_set]
        dep_testing_set = [td.objective for td in testing_set]

        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        CART = CART.fit(indep_training_set, dep_training_set)

        predictions = [float(x) for x in CART.predict(indep_testing_set)]
        mre = []
        for i, j in zip(dep_testing_set, predictions):
            mre.append(abs(i - j) / float(i))

        from numpy import mean, median, std
        output_string += str(len(training_set)) + "|"+ str(mean(mre))+"|"+ str(median(mre))+"|"+ str(std(mre)) + "\n"

    # File operation
    f = open(output_file, "w")
    f.write(output_string)
    f.close()


def get_elbow_point(dataset_name):
    data = read_csv(dataset_name)
    from sklearn.cluster import KMeans
    decision_data = [d.decisions for d in data]
    elbow_distance = []
    for i in xrange(1, 2*len(decision_data[0])):
        K = KMeans(n_clusters=i)
        y_pred = K.fit_predict(decision_data)
        elbow_distance.append(K.inertia_)

    # plot
    import matplotlib.pyplot as plt
    s = range(1, 2*len(decision_data[0]))
    plt.plot(s, elbow_distance)

    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of distance')
    plt.title(dataset_name)
    plt.grid(True)
    filename = "./Data/output_elbow_point/" + dataset_name + ".png"
    plt.savefig(filename)
    plt.cla()


def generate_graph(dataset_name, cluster_numbers):
    from os import system
    system("mkdir ./Data/output_knn_hotspot/" + dataset_name + "/graph/")
    graph_name = "./Data/output_knn_hotspot/" + dataset_name + "/graph/" + dataset_name + "_" + str(cluster_numbers) + ".png"

    filename = "./Data/output_knn_hotspot/" + dataset_name + "/data/" + dataset_name + "_" + str(cluster_numbers)
    contents = open(filename, "r").readlines()
    contents = [map(float, content.split("|")) for content in contents]
    plt.plot([c[0] for c in contents], [c[1] for c in contents], c="green", label="mean")
    plt.plot([c[0] for c in contents], [c[2] for c in contents], c="blue", label="median")
    plt.plot([c[0] for c in contents], [c[3] for c in contents], c="red", label="std")
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(graph_name)
    plt.cla()

if __name__ == "__main__":
    datasets = [ 'llvm', 'sqlite', 'x264']
    for dataset in datasets:
        for cluster in xrange(2, 8):
            run_experiment(dataset, cluster)
            filename = "./Data/output/" + dataset + "/" + dataset + "_" + str(cluster)
            generate_graph(dataset, cluster)

        # print "- " * 50
        # get_elbow_point(dataset)