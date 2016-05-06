import sys


def WHEREDataTransformation(df):
    from Utilities.WHERE.where import where
    # The Data has to be access using this attribute table._rows.cells

    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    clusters = where(data)

    return clusters


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
        data.append(data_item(i, [1 if x == "1" else 0 for x in row[:-1]], float(row[-1]) * (10**4))) # TODO: DecisionTree regressor returns int values. As a work around I multiply all the class values by 10**4
    f.close()
    if header is True: return H, data
    return data


def create_condition(df, list):
    condi = True
    header = df.columns
    assert(len(header) == len(list)), "something is wrong"
    for h,l in zip(header, list):
        condi = condi & df[h] == l
    return condi


def run_experiment_100(filename, count_member=None):
    from random import choice
    import numpy as np

    import pandas as pd
    df = pd.read_csv(filename)

    training_data = df

    independent_columns = [h for h in df.columns if '$<' not in h]
    dependent_columns = [h for h in df.columns if '$<' in h]

    full_training_independent = df[independent_columns]
    full_training_dependent = df[dependent_columns]

    data_dict = {}
    for iti in xrange(full_training_independent.shape[0]):
        row = full_training_independent.values[iti].tolist()
        key = "[" + ",".join(map(str, row)) + "]"
        data_dict[key] = full_training_dependent.values[iti].tolist()

    clusters = WHEREDataTransformation(training_data)
    print "Finished WHERE"


    testing_independent = []
    sampled_training_data_independent = []
    for c in clusters:
        temp = choice(range(len(c)))
        sampled_training_data_independent.append(c[temp])
        c.pop(temp)
        testing_independent.extend(c)

    sampled_training_data_dependent = []
    for stdi in sampled_training_data_independent:
        key = "[" + ",".join(map(str, stdi)) + "]"
        sampled_training_data_dependent.append(data_dict[key])

    testing_dependent = []
    for tdi in testing_independent:
        key = "[" + ",".join(map(str, tdi)) + "]"
        testing_dependent.append(data_dict[key])


    mres = []
    for i in xrange(3):
        t_sampled_training_data_dependent = [s[i] for s in sampled_training_data_dependent]
        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        CART = CART.fit(sampled_training_data_independent, t_sampled_training_data_dependent)

        t_testing_dependent = [t[i] for t in testing_dependent]
        predictions = [float(x) for x in CART.predict(testing_independent)]

        mre = []
        for i, j in zip(t_testing_dependent, predictions):
            mre.append(abs(i - j) / float(i))


        from numpy import mean, std
        mres.append(round(mean(mre), 5)*100)
    print mres
    return mres


def run_experiment_40(filename, count_member=None):
    from random import choice
    import numpy as np

    import pandas as pd
    df = pd.read_csv(filename)

    indexes = np.random.rand(df.shape[0]) < 0.4

    training_data = df[indexes]
    testing_data = df[~indexes]

    independent_columns = [h for h in df.columns if '$<' not in h]
    dependent_columns = [h for h in df.columns if '$<' in h]
    assert(len(dependent_columns) == 1), "Something is wrong"

    full_training_independent = training_data[independent_columns]
    full_training_dependent = training_data[dependent_columns]

    data_dict = {}
    for iti in xrange(full_training_independent.shape[0]):
        row = full_training_independent.values[iti].tolist()
        key = "[" + ",".join(map(str, row)) + "]"
        data_dict[key] = full_training_dependent.values[iti].tolist()

    testing_independent = testing_data[independent_columns]
    testing_dependent = testing_data[dependent_columns]

    clusters = WHEREDataTransformation(training_data)

    sampled_training_data_independent = [choice(c) for c in clusters]
    assert(len(sampled_training_data_independent) == len(clusters)), "something is wrong"

    sampled_training_data_dependent = []
    for stdi in sampled_training_data_independent:
        key = "[" + ",".join(map(str, stdi)) + "]"
        sampled_training_data_dependent.append(data_dict[key])

    from sklearn import tree
    CART = tree.DecisionTreeRegressor()
    CART = CART.fit(sampled_training_data_independent, sampled_training_data_dependent)

    predictions = [float(x) for x in CART.predict(testing_independent)]

    mre = []
    for i, j in zip(testing_dependent[dependent_columns[-1]].tolist(), predictions):
        mre.append(abs(i - j) / float(i))

    from numpy import mean
    return round(mean(mre), 5), len(sampled_training_data_dependent)


def experiment_what_so1(filename):
    mres = []
    evals = []
    print filename
    for _ in xrange(10):
        mre, eval = run_experiment_40(filename)
        mres.append(mre)
        evals.append(eval)
        print filename, mre*100, eval
        sys.stdout.flush()

    import pickle
    pickle.dump( mres, open( filename+"_mres.p", "wb" ))
    pickle.dump( evals, open( filename+"_evals.p", "wb" ))

    from numpy import mean, std
    print mean(mres), round(std(mres))


folder_name = "./Data/input/"
# so_filenames = ["AJStats.csv",
#                 "Apache.csv", "BerkeleyC.csv", "BerkeleyDB.csv", "BerkeleyDBC.csv", "BerkeleyDBJ.csv",
#                 "clasp.csv", "Dune.csv", "Elevator.csv", "Email.csv", "EPL.csv", "Hipacc.csv", "JavaGC.csv", "LinkedList.csv",
#                 "lrzip.csv", "PKJab.csv", "PrevaylerPP.csv", "SQLite.csv", "Wget.csv", "x264", "mo_ZipMe.csv"]

so_filenames = ["mo_ZipMe.csv"]
for filename in so_filenames:
    experiment_what_so1(folder_name + filename)