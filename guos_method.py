from __future__ import division
import os, sys, math
import matplotlib.pyplot as plt
from random import choice, randint, shuffle
from numpy import mean
import pandas as pd
import decimal
decimal.setcontext(decimal.Context(prec=34))




def run_experiment(dataset_name):

    def get_error(training_indep, training_dep, testing_indep, testing_dep):
        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        CART.fit(training_indep, training_dep)
        predictions = [float(x) for x in CART.predict(testing_indep)]
        mre = []
        for i, j in zip(testing_dep, predictions): mre.append(abs(i - j) / float(i))
        return mean(mre)


    df = pd.read_csv("./Data/normalized_input/normalized_" + dataset_name)
    headers = [h for h in df.columns if '$<' not in h]
    dependents = [h for h in df.columns if '$<' in h]
    data = df[headers]
    """ Get all the scores """
    scores = {}
    for i in xrange(len(df)):
        key = ",".join(map(str, df[headers].iloc[i]))
        scores[key] = df[dependents].iloc[i][-1]

    """ Find validation set"""
    indexes = range(len(df))
    shuffle(indexes)

    """ Test Data Sizes """
    training_set_size = int(len(indexes) * 0.4)
    validation_set_size = 30
    testing_set_size = len(indexes) - training_set_size - validation_set_size
    initial_training_size = 10

    training_indexes = indexes[:training_set_size]
    validation_indexes = indexes[training_set_size:training_set_size+30]
    test_indexes = indexes[training_set_size+30:]
    """ Select training set and increase it in steps of 10 """

    initial_training_indexes = []
    for _ in xrange(initial_training_size):
        initial_training_indexes.append(choice(training_indexes))
        """Removing the indexes from the training Indexes"""
        training_indexes.remove(initial_training_indexes[-1])

    assert(len(set(initial_training_indexes)) == len(initial_training_indexes)), "Making sure none of the trianing points are repeated"

    """ For initial training data"""
    training_indep = []
    training_dep = []
    for iti in initial_training_indexes:
        temp_holder = data.iloc[iti].tolist()
        training_indep.append(temp_holder)
        """ Finding the depended values of training Indep"""
        key = ",".join(map(str, temp_holder))
        training_dep.append(scores[key])
    assert(len(training_dep) == len(training_indep)), "The length of training and testing should be the same"

    """ For validation data"""
    validation_indep = []
    validation_dep = []
    for vi in validation_indexes:
        temp_holder = data.iloc[vi]
        validation_indep.append(temp_holder)
        """ Finding the depended values of training Indep"""
        key = ",".join(map(str, temp_holder))
        validation_dep.append(scores[key])
    assert (len(validation_indep) == len(validation_dep)), "The length of training and testing should be the same"

    """ For Testing data """
    testing_indep = []
    testing_dep = []
    for vi in test_indexes:
        temp_holder = data.iloc[vi]
        testing_indep.append(temp_holder)
        """ Finding the depended values of training Indep"""
        key = ",".join(map(str, temp_holder))
        testing_dep.append(scores[key])
    assert (len(testing_indep) == len(testing_dep)), "The length of training and testing should be the same"


    error = get_error(training_indep, training_dep, validation_indep, validation_dep)

    while error > 0.05 and len(training_dep) < training_set_size:
        new_point = choice(training_indexes)
        """ Removing the point from the indexes"""
        training_indexes.remove(new_point)
        new_point_indep = data.iloc[new_point]
        key = ",".join(map(str, new_point_indep))
        new_point_dep = scores[key]

        training_indep.append(new_point_indep)
        training_dep.append(new_point_dep)

        error = get_error(training_indep, training_dep, validation_indep, validation_dep)

    assert(len(training_dep) == len(training_indep)), "Something is wrong"

    error = get_error(training_indep, training_dep, testing_indep, testing_dep)

    return error, len(training_dep)

if __name__ == "__main__":
    import sys
    from random import seed
    seed(10)

    print "Note: This code is used to randomly sample the points in the training set"

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
