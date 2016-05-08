datasets = [    "Apache.csv", "BerkeleyC.csv", "BerkeleyDB.csv", "BerkeleyDBC.csv", "BerkeleyDBJ.csv",
                    "clasp.csv", "Dune.csv", "EPL.csv", "Hipacc.csv", "JavaGC.csv", "LinkedList.csv",
                    "lrzip.csv", "PKJab.csv", "SQLite.csv", "Wget.csv", "AJStats.csv"]

for dataset in datasets:
    folder_name = "./input/"
    eval_filename = folder_name + dataset + "_evals.p"
    mre_filename = folder_name + dataset + "_mres.p"

    import pickle
    evals = pickle.load( open(eval_filename, "rb" ))
    mres = pickle.load(open(mre_filename, "rb"))

    import numpy as np
    print dataset, round(np.mean(mres)*100, 3), round(np.std(mres)*100, 3), round(np.mean(evals), 3), round(np.std(mres), 3)
