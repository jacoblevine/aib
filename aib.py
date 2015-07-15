import itertools
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score as mutual_information


class DataPoint:
    """
    Wrapper for data point that carries three pieces of information:
    1) Actual data location
    2) Relevance variable value
    3) Index in data set
        The index is a class variable that is incremented at each new creation of a DataPoint object
        (i.e., DataPoint objects need to be created in order for the index to be correct)
    """
    index = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.index = DataPoint.index
        DataPoint.index += 1


class ClusterDict(dict):

    @property
    def n(self):
        return sum(len(t) for t in self.values())

    @property
    def Y(self):
        return set(c.y for t in self.values() for c in t)

    def calc_merge_cost(self, i, j):
        zi = self[i]
        zj = self[j]
        n = self.n
        Y = self.Y
        pzi = len(zi) / n
        pzj = len(zj) / n
        pzk = pzi + pzj
        pi = {i: pzi/pzk, j: pzj/pzk}
        bins = list(Y) + [max(Y) + 1]
        d = {i: np.histogram([c.y for c in zi], bins, normed=True)[0],
             j: np.histogram([c.y for c in zj], bins, normed=True)[0]}
        jsd = entropy(sum(pi[q] * d[q] for q in [i, j])) - sum(pi[q] * entropy(d[q]) for q in [i, j])
        return pzk * jsd

    def merge(self, i, j):
        target = min([i, j])
        self[target] = self[i] + self[j]
        remove = max([i, j])
        del self[remove]


class Partition:

    clusters = ClusterDict()
    merge_costs = dict()

    def __init__(self, data, relevance_variable):
        self.n = data.shape[0]
        self.Y = len(set(relevance_variable))
        """Construct dataset as a numpy array of DataPoint objects
        By processing data in order, each DataPoint carries with it its index in the data set
        A numpy array is used to support boolean indexing in the for loop below"""
        dataset = np.array([DataPoint(x, y) for x, y in zip(data, relevance_variable)])
        for i, x in enumerate(set(data)):
            index = data == x
            self.clusters[i] = tuple(dataset[index])
        self.calc_all_merge_costs()

    @property
    def m(self):
        return len(self.clusters)

    @property
    def assignments(self):
        """Iterate through ClusterDict and return cluster assignments, sorted by data index"""
        i, z = zip(*sorted([(point.index, key) for key, datapoints in self.clusters.items() for point in datapoints]))
        return z

    # @property
    # def xbar(self):
    #     assignments = [(c.x, c.)]

    def calc_all_merge_costs(self):
        pairs = itertools.combinations([k for k in self.clusters.keys()], 2)
        for pair in pairs:
            self.merge_costs[pair] = self.clusters.calc_merge_cost(*pair)

    def cluster_distance(self, i, j):
        xi = np.mean([c.x for c in self.clusters[i]])
        xj = np.mean([c.x for c in self.clusters[j]])
        return np.sqrt((xi - xj) ** 2)

    def find_merge_pair(self):
        min_pair = min(self.merge_costs, key=lambda x: self.merge_costs[x])
        min_val = self.merge_costs[min_pair]
        ties = [k for k, v in self.merge_costs.items() if v == min_val]
        if len(ties) > 1:
            d = {pair: self.cluster_distance(*pair) for pair in ties}
            min_pair = min(d, key=lambda x: d[x])
            assert(self.merge_costs[min_pair] == min_val)
        return min_pair

    def merge_next(self):
        min_pair = self.find_merge_pair()
        self.clusters.merge(*min_pair)
        remove = [key for key in self.merge_costs.keys()
                  if any([k in min_pair for k in key])]
        self.merge_costs = {key: value for key, value in self.merge_costs.items()
                            if key not in remove}
        self.calc_all_merge_costs()


def aib(data, relevance_variable):
    z = Partition(data, relevance_variable)
    result = {z.m: z.assignments}
    score = {z.m: mutual_information(z.assignments, relevance_variable)}
    while z.m > 1:
        z.merge_next()
        m = z.m
        d = z.assignments
        result[m] = d
        score[m] = mutual_information(d, relevance_variable)
    return result, score
