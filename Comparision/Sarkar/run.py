def get_results(training_set, testing_set):
    from sklearn import tree

    indep_training_set = [td.decisions for td in training_set]
    dep_training_set = [td.objective for td in training_set]

    CART = tree.DecisionTreeRegressor()
    CART = CART.fit(indep_training_set, dep_training_set)

    indep_testing_set = [td.decisions for td in testing_set]
    dep_testing_set = [td.objective for td in testing_set]

    predictions = [float(x) for x in CART.predict(indep_testing_set)]
    mre = []
    for i, j in zip(dep_testing_set, predictions):
        mre.append(abs(i - j) / float(i))

    from numpy import mean
    return mean(mre)

class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)


def read_csv(filename, header=False):
    def transform(filename):
        return "./data/raw_input/" + filename

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


if __name__ == "__main__":
    measurements = {"ajstats":2013, "Apache":55, "BerkeleyC":219, "BerkeleyDB":97, "BerkeleyDBC":161, "BerkeleyDBJ":57, "clasp":167, "EPL":104, "lrzip":47}
    datasets = ["ajstats", "Apache", "BerkeleyC", "BerkeleyDB", "BerkeleyDBC", "BerkeleyDBJ", "clasp", "EPL", "lrzip"]
    repeat = 10

    for dataset in datasets:
        data = read_csv(dataset)

        mean_list = []

        for _ in xrange(repeat):
            indexes = range(len(data))

            from random import shuffle
            shuffle(indexes)

            training_set_indices = indexes[:int(len(indexes) * 0.4)]
            training_set = [data[i] for i in training_set_indices]
            sarkar_training_set = training_set[:measurements[dataset]]

            testing_set_indices = indexes[int(len(indexes) * 0.4):]
            testing_set = [data[i] for i in testing_set_indices]

            mean_list.append(get_results(training_set, testing_set))

        from numpy import mean, std
        print "Dataset: ", dataset," Mean: ", round(mean(mean_list)*100, 3), " Std: ", round(std(mean_list), 3),
        print "#Evals: ", measurements[dataset]

