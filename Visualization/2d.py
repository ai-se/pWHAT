from __future__ import division
import pandas as pd
import numpy as np
import math
import os
import sys
from random import shuffle


def read_csv(filename, header=False):
    def transform(filename):
        return "../Data/normalized_input/normalized_" + filename

    import csv
    data = []
    dependent = []
    print transform(filename)
    f = open(transform(filename), 'rb')
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if i == 0 and header is False: continue  # Header
        elif i ==0 and header is True:
            H = row
            continue
        data.append([float(x) for x in row[:-1]])
        dependent.append(float(row[-1]))
    f.close()
    if header is True: return H, data
    return data, dependent


def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance


def equal_list(lista, listb):
    """Checks whether two list are same"""
    assert (len(lista) == len(listb)), "Not a valid comparison"
    for i, j in zip(lista, listb):
        if i == j:
            pass
        else:
            return False
    return True

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
    print "Started comparisions"
    for i in xrange(len(data)):
        print "#"
        for j in xrange(i, len(data)):
            print ".",
            if i == j: distance_matrix[i][j] = 0
            elif distance_matrix[i][j] == -1:
                distance_matrix[i][j] = euclidean_distance(data[i], data[j])
                if distance_matrix[i][j] > max_value:
                    max_value = distance_matrix[i][j]
                    max_i = i
                    max_j = j
        sys.stdout.flush()
    print "Finished comparisions"
    return data[max_i], data[max_j]

def spectral_dimensions(just_decisions):

    def x_form(a,b,c): return (a**2 + c**2 - b**2)/(2*c)

    def y_form(a, xx):
        if a - xx < 1e-6: return 0
        return (a**2 - xx**2)**0.5



    east, west = comparision_2n(just_decisions)
    c = euclidean_distance(east, west)

    spectral_d = []
    for d in just_decisions:
        a = euclidean_distance(east, d)
        b = euclidean_distance(west, d)
        x = x_form(a,b,c)
        y = y_form(a, x)
        spectral_d.append([x, y])

    return spectral_d


def two_dimension(dataset_file):
    contents, dependent = read_csv(dataset_file)
    spectral_d = spectral_dimensions(contents)

    import itertools
    spectral_d.sort()
    set_spectral_d = list(spectral_d for spectral_d, _ in itertools.groupby(spectral_d))

    spectrum_dict = {}
    spectrum_scores = {}
    for i, sd in enumerate(spectral_d):
        key = str(sd[0]) + "," + str(sd[1])
        if key in spectrum_dict.keys():
            spectrum_dict[key] += 1
            spectrum_scores[key].append(dependent[i])
        else:
            spectrum_dict[key] = 1
            spectrum_scores[key] = [dependent[i]]


    for ke in spectrum_scores.keys():
        print ke, spectrum_dict[ke], np.mean(spectrum_scores[ke]), np.std(spectrum_scores[ke])

    weights = []
    for ssd in set_spectral_d:
        key = str(ssd[0]) + "," + str(ssd[1])
        weights.append(spectrum_dict[key])

    n_weight = [(w - min(weights))/(max(weights) - min(weights)) for w in weights]

    x = [sd[0] for sd in spectral_d]
    y = [sd[1] for sd in spectral_d]

    import matplotlib.pyplot
    import pylab

    matplotlib.pyplot.scatter(x, y, s=n_weight, cmap='gray')

    matplotlib.pyplot.savefig("./Output/" + dataset_file[:-3])
    matplotlib.pyplot.cla()







if __name__ == "__main__":
    import sys


    datasets = ["AJStats.csv", "Apache.csv", "BerkeleyC.csv", "BerkeleyDB.csv", "BerkeleyDBC.csv", "BerkeleyDBJ.csv",
                    "clasp.csv", "Dune.csv",
                     "EPL.csv", "Hipacc.csv", "LinkedList.csv",
                     "lrzip.csv", "PKJab.csv", "SQLite.csv", "Wget.csv", "x264.csv", "ZipMe.csv"]
    # datasets = ["Dune.csv"]#, "Hipacc.csv", "JavaGC.csv"]

    # datasets = ["AJStats.csv"]
    for dataset in datasets:
        two_dimension(dataset)