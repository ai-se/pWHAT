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
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)


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


def furthest(one, all_members):
        """Find the distant point (from the population) from one (point)"""
        ret = None
        ret_distance = -1 * 1e10
        for member in all_members:
            if equal_list(one, member) is True:
                continue
            else:
                temp = euclidean_distance(one, member)
                if temp > ret_distance:
                    ret = member
                    ret_distance = temp
        return ret


def equal_list(lista, listb):
        """Checks whether two list are same"""
        assert (len(lista) == len(listb)), "Not a valid comparison"
        for i, j in zip(lista, listb):
            if i == j:
                pass
            else:
                return False
        return True


def comparision_2n(data):
    from random import choice
    any_point = choice(data)
    east = furthest(any_point, data)
    west = furthest(east, data)
    return east, west


def comparision_n2(data):
    distance_matrix = [[-1 for _ in xrange(len(data))] for _ in xrange(len(data))]
    max_value = -1e10
    max_i = -1
    max_j = -1
    for i in xrange(len(data)):
        for j in xrange(len(data)):
            if i == j: distance_matrix[i][j] = 0
            elif distance_matrix[i][j] == -1:
                distance_matrix[i][j] = euclidean_distance(data[i], data[j])
                if distance_matrix[i][j] > max_value:
                    max_value = distance_matrix[i][j]
                    max_i = i
                    max_j = j
    return data[max_i], data[max_j]


def generate_hist(data):
    members = sorted(set([d.hotspot_scores for d in data]), reverse=True)
    d = {}
    for member in members:
        d[str(member)] = [dd.id for i, dd in enumerate(data) if dd.hotspot_scores == member]
    return d


def get_indices(data, count_members):
    indices_dict = generate_hist(data)
    return_indices = []
    for key in sorted(indices_dict.keys(), reverse=True):
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


def find_hotspot(data, count_members):

    return get_indices(data, count_members)


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
        data.append(data_item(i, [1 if x == "Y" else 0 for x in row[:-1]], float(row[-1]) * (10**4))) # TODO: DecisionTree regressor returns int values. As a work around I multiply all the class values by 10**4
    f.close()
    if header is True: return H, data
    return data


def add_spectral_dimensions(data):

    def x_form(a,b,c): return (a**2 + c**2 - b**2)/(2*c)

    def y_form(a, xx):
        if a - xx < 1e-6: return 0
        return (a**2 - xx**2)**0.5

    just_decisions = [d.decisions for d in data]

    east, west = comparision_n2(just_decisions)
    c = euclidean_distance(east, west)

    for d in data:
        a = euclidean_distance(east, d.decisions)
        b = euclidean_distance(west, d.decisions)
        x = x_form(a,b,c)
        y = y_form(a, x)
        d.spectral_decisions = [x, y]

    return data


def get_hotspot_scores(data):
    distance_matrix = [[-1 for _ in xrange(len(data))] for _ in xrange(len(data))]
    from sklearn.metrics import jaccard_similarity_score
    for i in xrange(len(data)):
        for j in xrange(len(data)):
            if distance_matrix[i][j] == -1 and i != j:
                # print data[i].spectral_decisions, data[j].spectral_decisions
                # print data[i].decisions, data[j].decisions
                distance_matrix[i][j] = euclidean_distance(data[i].spectral_decisions, data[j].spectral_decisions)
                distance_matrix[j][i] = distance_matrix[i][j]
            elif distance_matrix[i][j] == -1 and i == j:
                distance_matrix[j][i] = 0
            else:
                pass

    hotspot_scores = []
    for i in xrange(len(data)):
        data[i].hotspot_scores = sum([1/(distance_matrix[i][j]**0.5) for j in xrange(len(distance_matrix)) if distance_matrix[i][j] != 0])

    # [sum([1/data[i][j] for j in xrange(len(data))]) for i in xrange(len(data))]
    # print "Done calculating hotspot scores"
    return data


def run_experiment_3(dataset_name, count_member=None):
    # read raw data
    raw_data = read_csv(dataset_name)

    # add x, y dimensions
    spectral_data = add_spectral_dimensions(raw_data)
    assert(len(spectral_data) == len(raw_data)), "Something is wrong"

    for d in spectral_data:
        print d.spectral_decisions

    exit()

    from random import shuffle
    indexes = range(len(spectral_data))
    shuffle(indexes)
    # getting the reserve testing data
    testing_reserve_indexes = indexes[:30]
    testing_reserve = [spectral_data[i] for i in testing_reserve_indexes]

    # getting the rest of the data
    from copy import deepcopy
    data = deepcopy(spectral_data)
    for i in sorted(testing_reserve_indexes, reverse=True): data.pop(i)

    assert(len(data) + len(testing_reserve) == len(raw_data)), "something is wrong"

    # adding hotspot scores to all the data points
    data = get_hotspot_scores(data)

    # converting to a dict for easy access
    data_dict = {}
    for d in data:
        assert((d.id in data_dict.keys()) is False), "Something is wrong"
        data_dict[d.id] = d

    fault_rate = 1
    count_member = 0
    while fault_rate > .03 and count_member < 0.4 * len(data):
        training_indices = find_hotspot(data, count_member+1)
        training_set = [data_dict[i] for i in training_indices]
        indep_training_set = [td.decisions for td in training_set]
        dep_training_set = [td.objective for td in training_set]

        indep_testing_set = [td.decisions for td in testing_reserve]
        dep_testing_set = [td.objective for td in testing_reserve]

        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        CART = CART.fit(indep_training_set, dep_training_set)

        predictions = [float(x) for x in CART.predict(indep_testing_set)]
        mre = []
        for i, j in zip(dep_testing_set, predictions):
            mre.append(abs(i - j) / float(i))

        from numpy import mean
        fault_rate = mean(mre)
        count_member += 1

    data_indices = [d.id for d in data]
    for index in training_indices:
        data_indices.remove(index)
    assert(len(data_indices) + len(training_indices) == len(data)), "something is wrong"

    testing_set = [data_dict[i] for i in data_indices]

    CART = tree.DecisionTreeRegressor()
    CART = CART.fit(indep_training_set, dep_training_set)

    indep_testing_set = [td.decisions for td in testing_set]
    dep_testing_set = [td.objective for td in testing_set]

    predictions = [float(x) for x in CART.predict(indep_testing_set)]
    mre = []
    for i, j in zip(dep_testing_set, predictions):
        mre.append(abs(i - j) / float(i))

    from numpy import mean
    return mean(mre), len(indep_training_set)


if __name__ == "__main__":
    from random import seed

    from numpy import mean, median, std
    datasets = [ ['bc', 64],]
    for dataset in datasets:
        scores = []
        counts = []
        seed(10)
        for _ in xrange(10):
            print ". ",
            import sys
            sys.stdout.flush()
            mre, count = run_experiment_3(dataset[0], dataset[1])
            scores.append(mre)
            counts.append(count)
        print dataset[0], " | ", mean(scores) * 100, " | ", std(scores)*100,
        print counts
        print "- " * 50