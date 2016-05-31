from __future__ import division
import os, sys, math
import matplotlib.pyplot as plt
from random import choice, randint
from numpy import mean
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
        for i, member in enumerate(all_members):
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
        for j in xrange(i, len(data)):
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
        return "./Data/normalized_input/normalized_" + filename

    import csv
    data = []
    print transform(filename)
    f = open(transform(filename), 'rb')
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if i == 0 and header is False: continue  # Header
        elif i ==0 and header is True:
            H = row
            continue
        data.append(data_item(i, [float(x) for x in row[:-1]], float(row[-1]) * (10**4))) # TODO: DecisionTree regressor returns int values. As a work around I multiply all the class values by 10**4
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
        for j in xrange(i, len(data)):
            if distance_matrix[i][j] == -1 and i != j:
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
    return sorted(data, key=lambda x: x.hotspot_scores, reverse=True)


def WHEREDataTransformation(filename):
    from Utilities.WHERE.where import where_orginal
    # The Data has to be access using this attribute table._rows.cells
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    dependents = [h for h in df.columns if '$<' in h]

    scores = {}
    for i in xrange(len(df)):
        key = ",".join(map(str, df[headers].iloc[i]))
        scores[key] = df[dependents].iloc[i][-1]

    data = df[headers]
    clusters = where_orginal(data)

    return clusters, scores


def sorting_function(list_data):
    hotspot_scores = [ld.hotspot_scores for ld in list_data]
    from numpy import percentile
    return percentile(hotspot_scores, 75) - percentile(hotspot_scores, 25)


def return_content(header, content):
    return_c = ",".join(header) + "\n"
    for c in content:
        return_c += ",".join(map(str, c.decisions)) + "," + str(c.objective) + "\n"

    return return_c


def aDist(one, two):
    return euclidean_distance(one.tolist(), two.tolist())


def farthest(one, rest):
    return sorted(rest, key=lambda F: aDist(F, one))[-1]

def find_variance_ranks(rest_indep):
    # Calculate Variances in the Clusters
    variance_scores = []
    for rest_i in rest_indep:
        """ This is to avoid cases when the cluster is empty """
        if len(rest_i) == 0:
            variance_scores.append(-1)
            continue
        random_point = rest_i[randint(0, len(rest_i) - 1)]
        one = farthest(random_point, rest_i)
        two = farthest(one, rest_i)
        variance_scores.append(aDist(one, two))
    variances_ranks = [i[0] for i in sorted(enumerate(variance_scores), reverse=True, key=lambda x:x[1])]
    return variances_ranks

def run_experiment(dataset_name):
    size_of_commitee = 10

    def get_error(training_indep, training_dep, testing_indep, testing_dep):
        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        CART.fit(training_indep, training_dep)
        predictions = [float(x) for x in CART.predict(testing_indep)]
        mre = []
        for i, j in zip(testing_dep, predictions): mre.append(abs(i - j) / float(i))
        return mean(mre)

    def get_prediction(Kart, testing_indep):
        predictions = [float(x) for x in Kart.predict(testing_indep)]
        assert (len(predictions) == 1), "Should only predict one value"
        return predictions[-1]

    def generate_commitee(training_indep, training_dep):
        CARTS = []
        committee_indexes = [[randint(0, len(training_indep) - 1) for _ in xrange(len(training_indep))] for _ in
                             xrange(size_of_commitee)]
        for committee_index in committee_indexes:
            temp_training_indep = [training_indep[i] for i in committee_index]
            temp_training_dep = [training_dep[i] for i in committee_index]
            from sklearn import tree
            CART = tree.DecisionTreeRegressor()
            CARTS.append(CART.fit(temp_training_indep, temp_training_dep))
        return CARTS

    clusters, scores = WHEREDataTransformation("./Data/normalized_input/normalized_" + dataset_name)

    training_indep = []
    validation_indep = []
    rest_indep = []
    for cluster in clusters:
        one_random_point = randint(0, len(cluster)-1)
        two_random_point = randint(0, len(cluster)-1)
        while one_random_point == two_random_point: two_random_point = randint(0, len(cluster)-1)

        rest_point = range(len(cluster))
        rest_point.remove(one_random_point)
        rest_point.remove(two_random_point)

        training_indep.append(cluster[one_random_point])
        validation_indep.append(cluster[two_random_point])
        rest_indep.append([cluster[i] for i in rest_point])

    # get testing_data
    keys = scores.keys()
    # # remove keys of training_data from variable "keys"
    training_keys = [",".join(map(str, td.tolist())) for td in training_indep]
    training_dep = [scores[training_key] for training_key in training_keys]
    for training_key in training_keys:keys.remove(training_key)
    assert (len(keys) + len(training_keys) == len(scores.keys())), "somethign is wrong"

    validation_keys = [",".join(map(str, vi.tolist())) for vi in validation_indep]
    validation_dep = [scores[validation_key] for validation_key in validation_keys]
    for validation_key in validation_keys:keys.remove(validation_key)
    assert (len(keys) + len(training_keys)  + len(validation_keys) == len(scores.keys())), "somethign is wrong"

    rest_dep = []
    for cluster in rest_indep:
        rest_keys = [",".join(map(str, td.tolist())) for td in cluster]
        temp_rest_dep = [scores[rest_key] for rest_key in rest_keys]
        for rest_key in rest_keys: keys.remove(rest_key)
        rest_dep.append(temp_rest_dep)
    assert(len(keys) == 0), "It must be empty"


    # print "Length of training_samples : ", len(training_dep)

    """ This is to make sure that the loop terminates"""
    progressive_length = sum([len(ri) for ri in rest_indep])

    error_score = get_error(training_indep, training_dep, validation_indep, validation_dep)

    """ For debugging """
    passed_clusters = [0 for _ in xrange(len(rest_indep))]


    count = 0
    while error_score > 0.05 and count < progressive_length:
        """ Variance Ranks are calculated every interation so that it slowly converges """
        variance_ranks = find_variance_ranks(rest_indep)
        cluster_number = variance_ranks[0]
        try:
            cluster_index = choice(range(len(rest_indep[cluster_number])))
        except:
            import pdb
            pdb.set_trace()
            print count
            count += 1
            continue
        # print cluster_number, cluster_index

        """ Choose the independent and dependent value of the new point"""
        new_point_indep = rest_indep[cluster_number][cluster_index]
        new_point_dep = rest_dep[cluster_number][cluster_index]

        """ Ask commitee to vote for the new point"""
        predicted_values = []
        committee = generate_commitee(training_indep, training_dep)
        for member in committee: predicted_values.append(get_prediction(member, [new_point_indep]))

        """ Remove the index so that it can't be selected again"""
        rest_indep[cluster_number].pop(cluster_index)
        rest_dep[cluster_number].pop(cluster_index)
        assert(len(rest_indep) == len(rest_dep)), "The length of both rest_indep and rest_dep should be same"

        """ This is a proxy for entropy"""
        error_percentage = ((max(predicted_values) - min(predicted_values)) / min(predicted_values)) * 100
        if error_percentage > 15:
            training_indep.append(new_point_indep)
            training_dep.append(new_point_dep)
        else:
            """ This is to track which cluster is stable and which is not """
            passed_clusters[cluster_number] += 1
            # print passed_clusters

        error_score = get_error(training_indep, training_dep, validation_indep, validation_dep)
        count += 1

    """ To get testing data set
        - collects hash of all the elements
        - remove all the training elements
        - remove all the validation elements
        - what is remaining is the testing dataset
    """
    keys = scores.keys()
    # remove training_set
    training_keys = [",".join(map(str, td.tolist())) for td in training_indep]
    for training_key in training_keys: keys.remove(training_key)
    # remove validation_set
    validation_keys = [",".join(map(str, vi.tolist())) for vi in validation_indep]
    for validation_key in validation_keys: keys.remove(validation_key)


    testing_indep = []
    testing_dep = []
    for key in keys:
        testing_indep.append(map(float, key.split(",")))
        testing_dep.append(scores[key])

    error_rate = get_error(training_indep, training_dep, testing_indep, testing_dep)
    return error_rate, len(training_dep)

if __name__ == "__main__":
    import sys
    from random import seed
    seed(10)

    print "Note: This code uses the original WHERE code and doesn't use any pruning heuristics"

    datasets = ["Apache.csv", "BerkeleyC.csv", "BerkeleyDB.csv", "BerkeleyDBC.csv", "BerkeleyDBJ.csv",
                "clasp.csv", "Dune.csv", "EPL.csv", "LinkedList.csv",
                "lrzip.csv", "PKJab.csv", "SQLite.csv", "Wget.csv", "x264.csv", "ZipMe.csv", "AJStats.csv"]
    for dataset in datasets:
        mean_mre = []
        mean_length = []
        for _ in xrange(10):
            print ".",
            sys.stdout.flush()
            mre, datalength = run_experiment(dataset)
            mean_mre.append(mre)
            mean_length.append(datalength)
        from numpy import mean, std
        print
        print dataset, round(mean(mean_mre)*100, 3), round(std(mean_mre)*100, 3), mean(mean_length), round(std(mean_length), 3)
        print [round(r * 100, 3) for r in mean_mre]
        print [round(r, 3) for r in mean_length]
        print
