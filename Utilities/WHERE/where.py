from __future__ import division
from pdb import set_trace
import pandas as pd
import numpy as np
from os import walk
from random import randint as randi, seed as rseed
__author__ = 'vivekaxl'

def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance



def where(data, scores):
    """
    Recursive FASTMAP clustering.
    """
    if isinstance(data, pd.core.frame.DataFrame):
        data = data.as_matrix()
    if not isinstance(data, np.ndarray):
        raise TypeError('Incorrect data format. Must be a pandas Data Frame, or a numpy nd-array.')

    N = np.shape(data)[0]
    clusters = []
    norm = np.max(data, axis=0)[:-1] -np.min(data, axis=0)[:-1]

    def aDist(one, two):
        return euclidean_distance(one.tolist(), two.tolist())

    def farthest(one,rest):
        return sorted(rest, key=lambda F: aDist(F,one))[-1]

    def recurse(dataset, step=0):
        R, C = np.shape(dataset) # No. of Rows and Col
        # Find the two most distance points.
        one=dataset[randi(0,R-1)]
        mid=farthest(one, dataset)
        two=farthest(mid, dataset)

        mkey = ",".join(map(str, one.tolist()))
        tkey = ",".join(map(str, two.tolist()))

        mscore = scores[mkey]
        tscore = scores[tkey]

        if step >=2:
            heuristic = (abs(mscore-tscore)/min(mscore, tscore)) * 100
        else:
            heuristic = 100

        # Project each case on
        def proj(test):
            a = aDist(mid, test)
            b = aDist(two, test)
            c = aDist(mid, two)
            return (a**2-b**2+c**2)/(2*c)

        if R < np.sqrt(N) or heuristic <= 0.5: # since we need 64 cells
            clusters.append(dataset)
        else:
            _ = recurse(sorted(dataset,key=lambda F:proj(F))[:int(R/2)], step+1)
            _ = recurse(sorted(dataset,key=lambda F:proj(F))[int(R/2):], step+1)


    recurse(data)
    return clusters



def where_org(data):
    """
    Recursive FASTMAP clustering.
    """
    if isinstance(data, pd.core.frame.DataFrame):
        data = data.as_matrix()
    if not isinstance(data, np.ndarray):
        raise TypeError('Incorrect data format. Must be a pandas Data Frame, or a numpy nd-array.')

    N = np.shape(data)[0]
    clusters = []
    norm = np.max(data, axis=0)[:-1] -np.min(data, axis=0)[:-1]

    def aDist(one, two):
        return euclidean_distance(one.tolist(), two.tolist())

    def farthest(one,rest):
        return sorted(rest, key=lambda F: aDist(F,one))[-1]

    def recurse(dataset, step=0):
        R, C = np.shape(dataset) # No. of Rows and Col
        # Find the two most distance points.
        one=dataset[randi(0,R-1)]
        mid=farthest(one, dataset)
        two=farthest(mid, dataset)

        # Project each case on
        def proj(test):
            a = aDist(mid, test)
            b = aDist(two, test)
            c = aDist(mid, two)
            return (a**2-b**2+c**2)/(2*c)

        if R < np.sqrt(N): # since we need 64 cells
            clusters.append(dataset)
        else:
            _ = recurse(sorted(dataset,key=lambda F:proj(F))[:int(R/2)], step+1)
            _ = recurse(sorted(dataset,key=lambda F:proj(F))[int(R/2):], step+1)

    recurse(data)
    return clusters

def _test(dir='/Users/viveknair/GIT/CodeLab/SurrogateBuilder/Data/GALE4_DTLZ1_9_5/0/'):
    files=[]
    for (dirpath, _, filename) in walk(dir):
        for f in filename:

            df=pd.read_csv(dirpath+f)
            headers = [h for h in df.columns if '$<' not in h]
            files.append(df[headers])

    "For N files in a project, use 1 to N-1 as train."
    train = pd.concat(files[:-1])
    clusters = where(train)
    # ----- ::DEBUG:: -----
    set_trace()

if __name__=='__main__':
    _test()