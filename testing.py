from __future__ import division
import os, sys, math
import matplotlib.pyplot as plt
import numpy as np
import decimal
decimal.setcontext(decimal.Context(prec=34))


class data_item():
    def __init__(self, decisions, objective):
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


def find_hotspot(data, hotspot_scores, count_members):


    return get_indices(hotspot_scores, count_members)


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
        data.append(data_item([1 if x == "Y" else 0 for x in row[:-1]], float(row[-1]) * (10**4))) # TODO: DecisionTree regressor returns int values. As a work around I multiply all the class values by 10**4
    f.close()
    if header is True: return H, data
    return data


def get_hotspot_scores(data):
    distance_matrix = [[-1 for _ in xrange(len(data))] for _ in xrange(len(data))]
    from sklearn.metrics import jaccard_similarity_score
    for i in xrange(len(data)):
        for j in xrange(len(data)):
            if distance_matrix[i][j] == -1 and i != j:
                distance_matrix[i][j] = decimal.Decimal(jaccard_similarity_score(data[i].decisions, data[j].decisions))
                distance_matrix[j][i] = distance_matrix[i][j]
            elif distance_matrix[i][j] == -1 and i == j:
                distance_matrix[j][i] = 1
            else:
                pass
    hotspot_scores = [sum(distance_matrix[i]) for i in xrange(len(data))]
    print "Done calculating hotspot scores"
    return hotspot_scores


def run_experiment(dataset_name, count_members=None):
    output_file = "./Data/output/" + dataset_name
    data = read_csv(dataset_name)
    hotspot_scores = get_hotspot_scores(data)

    output_string = ""
    for count_member in range(int(len(data)* 0.2))[::100]:
        print ". ",
        sys.stdout.flush()
        training_indices =  find_hotspot(data, hotspot_scores, count_member+1)
        training_set = [data[i] for i in training_indices]
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


if __name__ == "__main__":
    datasets = [  'apache']
    for dataset in datasets:
        run_experiment(dataset)
        print "- " * 50