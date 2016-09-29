from __future__ import division

import decimal
import math
import pandas as pd
import numpy as np

decimal.setcontext(decimal.Context(prec=34))


class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id) + "|" + ",".join(map(str, self.decisions)) + "|" + str(self.objective)


def WHEREDataTransformation(df):
    from Utilities.WHERE.where import where_orginal
    headers = [h for h in df.columns if '$<' not in h]
    dependents = [h for h in df.columns if '$<' in h]

    scores = {}
    for i in xrange(len(df)):
        key = ",".join(map(str, df[headers].iloc[i]))
        scores[key] = df[dependents].iloc[i][-1]

    data = df[headers]
    clusters = where_orginal(data)

    return clusters, scores


def run_experiment(dataset_name):
    content = pd.read_csv("./Data/input/" + dataset_name)
    headers = [h for h in content.columns if '$<' not in h]
    dependents = [h for h in content.columns if '$<' in h][-1]
    mask = np.random.rand(len(content)) < 0.4

    train = content[mask]
    test = content[~mask]

    crossval = test.head(30)
    holdout = test.tail(-30)

    clusters, scores = WHEREDataTransformation(train)
    # print "Time taken for WHERE: ", (time.time() - begin_time) % 60

    from random import choice
    training_indep = [choice(c) for c in clusters]

    training_keys = [",".join(map(str, td.tolist())) for td in training_indep]
    training_dep = [scores[training_key] for training_key in training_keys]

    indep_testing_set = holdout[headers]
    dep_testing_set = holdout[dependents]

    assert (len(training_indep) == len(training_dep)), "Something is wrong"
    from sklearn import tree
    CART = tree.DecisionTreeRegressor()
    CART = CART.fit(training_indep, training_dep)

    predictions = [float(x) for x in CART.predict(indep_testing_set)]
    mre = []
    for i, j in zip(dep_testing_set.values.tolist(), predictions):
            mre.append(abs(i - j) / (float(i) + 0.00001))

    from numpy import mean
    return mean(mre), len(training_dep)



if __name__ == "__main__":
    import sys
    from random import seed

    seed(10)

    datasets = [ "Apache.csv", ]
    # "BerkeleyC.csv", "BerkeleyDB.csv", "BerkeleyDBC.csv", "BerkeleyDBJ.csv",
    #                 "clasp.csv", "Dune.csv", "EPL.csv", "Hipacc.csv", "LinkedList.csv",
    #                 "lrzip.csv", "PKJab.csv", "SQLite.csv", "Wget.csv", "x264.csv", "ZipMe.csv", "AJStats.csv"]

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
        print dataset, round(mean(mean_mre) * 100, 3), round(std(mean_mre) * 100, 3), mean(mean_length), std(
            mean_length)
        print [round(r * 100, 3) for r in mean_mre]
        print [round(r, 3) for r in mean_length]
        print
