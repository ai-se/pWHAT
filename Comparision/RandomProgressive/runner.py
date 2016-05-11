from __future__ import division
import pandas as pd
import numpy as np
import math
import os
import sys
from random import shuffle


def random_progressive_sampling(dataset_file):
    foldername = "../../Data/normalized_input/normalized_"
    dataset_name = foldername + dataset_file
    df = pd.read_csv(dataset_name)
    rows, columns = np.shape(df)  # No. of Rows and Col
    repeat = 10
    mres = []
    evals = []

    for _ in xrange(repeat):
        indexes = [i for i in xrange(rows)]
        shuffle(indexes)
        # getting the reserve testing data
        training_reservior_indexes = indexes[:int(rows * 0.4)]
        testing_reserve_indexes = indexes[int(rows * 0.4):int(rows * 0.4) + 30]
        testing_reservior_indexes = indexes[int(rows * 0.4) + 30:]

        training_data = df.iloc[training_reservior_indexes]
        testing_reserve = df.iloc[testing_reserve_indexes]
        testing_data = df.iloc[testing_reservior_indexes]

        assert(len(training_data) + len(testing_reserve) + len(testing_data) == rows), "Something is wrong"


        count = 10
        fault_rate = 1
        full_progressive_data_indexes = [i for i in xrange(len(training_data))]
        shuffle(full_progressive_data_indexes)

        while fault_rate > 0.07 and count < len(training_data) - 1:
            progressive_data_indexes = full_progressive_data_indexes[:count]
            progressive_training_data = training_data.iloc[progressive_data_indexes]

            indep_headers = [h for h in df.columns if '$<' not in h]
            dep_headers = [h for h in df.columns if '$<' in h]

            from sklearn import tree
            CART = tree.DecisionTreeRegressor()
            assert (len(progressive_training_data[indep_headers]) == len(progressive_training_data[dep_headers])), "something is wrong"
            CART = CART.fit(progressive_training_data[indep_headers], progressive_training_data[dep_headers])

            mre = []
            # print len(train[indep_headers]), len(test[indep_headers])
            predictions = [float(x) for x in CART.predict(testing_reserve[indep_headers])]
            for i, j in zip(testing_reserve[dep_headers[-1]].tolist(), predictions):
                if i != 0:
                    mre.append(abs(float(i) - j) / float(i))

            fault_rate = np.mean(mre)
            count += 1

        evals.append(count)
        final_data_indexes = full_progressive_data_indexes[:count]
        final_training_data = training_data.iloc[final_data_indexes]
        from sklearn import tree
        CART = tree.DecisionTreeRegressor()
        assert (len(final_training_data[indep_headers]) == len(final_training_data[dep_headers])), "something is wrong"

        CART = CART.fit(final_training_data[indep_headers], final_training_data[dep_headers])

        mre = []
        # print len(train[indep_headers]), len(test[indep_headers])
        predictions = [float(x) for x in CART.predict(testing_data[indep_headers])]
        for i, j in zip(testing_data[dep_headers[-1]].tolist(), predictions):
            if i != 0:
                mre.append(abs(float(i) - j) / float(i))

        mres.append(np.mean(mre)*100)
    print dataset_file, round(np.mean(mres), 3), round(np.std(mres), 3), round(np.mean(evals), 3), round(np.std(evals), 3)



if __name__ == "__main__":
    filenames = [file for file in os.listdir("../../Data/input/") if file != ".DS_Store" if file != "__init__.py"]
    from random import seed
    seed(10)
    for filename in filenames:
        random_progressive_sampling(filename)