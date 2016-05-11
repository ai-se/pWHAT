from __future__ import division
import pandas as pd
import numpy as np
import math
import os
import sys

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def Guo_Method(dataset_file):
    foldername = "../../Data/normalized_input/normalized_"
    dataset_name = foldername + dataset_file
    df = pd.read_csv(dataset_name)
    rows, columns = np.shape(df)  # No. of Rows and Col
    pair_wise_comparisions = nCr(columns, 2)

    temp_size = columns
    count = 0
    while temp_size < pair_wise_comparisions and temp_size < int(0.9 * rows):
        temp_size += columns
        count += 1
    sample_sizes = [i * columns for i in xrange(1, count + 1)]
    if pair_wise_comparisions < rows:
        sample_sizes += [pair_wise_comparisions]



    mres = []
    repeat = 10
    for sample_size in sample_sizes:
        repeats = []
        for _ in xrange(repeat):
            msk = np.random.rand(len(df)) < sample_size/rows
            train = df[msk]
            test = df[~msk]
            assert(len(train) + len(test) == rows), "Something is wrong"

            indep_headers = [h for h in df.columns if '$<' not in h]
            dep_headers = [h for h in df.columns if '$<' in h]

            from sklearn import tree
            CART = tree.DecisionTreeRegressor()
            assert(len(train[indep_headers]) == len(train[dep_headers])), "something is wrong"

            CART = CART.fit(train[indep_headers], train[dep_headers])

            mre = []
            # print len(train[indep_headers]), len(test[indep_headers])
            predictions = [float(x) for x in CART.predict(test[indep_headers])]
            for i, j in zip(test[dep_headers[-1]].tolist(), predictions):
                if i != 0:
                    mre.append(abs(float(i) - j) / float(i))
            repeats.append(np.mean(mre) * 100)
            # print ">> ", np.mean(mre)
        mres.append(np.mean(repeats))


        # print mres
        print dataset_file, sample_size, round(np.mean(repeats), 3), round(np.std(repeats), 3)
        sys.stdout.flush()

if __name__ == "__main__":
    filenames = [file for file in os.listdir("../../Data/input/") if file != ".DS_Store" if file != "__init__.py"]
    from random import seed
    seed(10)

    for filename in filenames:
        print filename + "---" * 10
        Guo_Method(filename)




