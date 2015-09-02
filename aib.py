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
    """
    Extension of native dict class, specialized to store a data set as clusters of DataPoint objects
    Each item in ClusterDict is a cluster, and the hash key is the cluster ID
    """

    def __init__(self):
        super().__init__()
        self._cached_Y = None


    @property
    def n(self):
        """Total number of DataPoints (not clusters!) in the ClusterDict"""
        return sum(len(t) for t in self.values())

    @property
    def Y(self):
        """Special feature for the AIB implementation:
        Return the set of all relevance_variable values
        (needed for calc_merge_cost)"""
        if self._cached_Y is None:
            return set(c.y for t in self.values() for c in t)
        else:
            return self._cached_Y

    @property
    def means(self):
        """Return a dictionary of hash keys with their cluster means"""
        return {key: np.mean([datapoint.x for datapoint in cluster]) for key, cluster in self.items()}

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
        target, remove = sorted([i, j])
        self[target] = self[i] + self[j]
        del self[remove]


class Partition:
    """
    Represent the current partition of the data into discrete categories

    Clusters are represented as tuples of DataPoint objects stored in a ClusterDict instance

    Partition implements selecting the pair of clusters to merge and sends the merge call to ClusterDict
    """

    def __init__(self, data, relevance_variable):
        """Initialize Partition, placing each unique value in data into its own cluster"""
        self.n = data.shape[0]
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        self.Y = len(set(relevance_variable))
        self.clusters = ClusterDict()
        self.merge_costs = dict()
        """Construct dataset as a numpy array of DataPoint objects
        By processing data in order, each DataPoint carries with it its index in the data set
        A numpy array is used to support boolean indexing in the for loop below
        """
        DataPoint.index = 0  # Every time a new partition is initialized, reset the DataPoint index tracker
        dataset = np.array([DataPoint(x, y) for x, y in zip(data, relevance_variable)])
        assert len(dataset) == DataPoint.index
        """
        If data are multidimensional, we need each entry as a tuple in order to hash
        """
        data_set = {tuple(row) for row in data}  # nb: data_set is an actual Set
        for i, x in enumerate(data_set):
            index = np.array([tuple(row) == x for row in data], dtype=bool)
            self.clusters[i] = tuple(dataset[index])
        self.calc_all_merge_costs()
        assert max([max(k) for k in self.merge_costs.keys()]) + 1 == self.m

    @property
    def m(self):
        """Number of clusters, m = |Z| (where Z is the Partition)"""
        return len(self.clusters)

    @property
    def assignments(self):
        """Iterate through ClusterDict and return cluster assignments, sorted by data index"""
        i, z = zip(*sorted([(point.index, key) for key, datapoints in self.clusters.items() for point in datapoints]))
        # Rename cluster labels (z values) so they are sorted by increasing cluster mean
        cluster_means = self.clusters.means
        rename = {j: i for i, j in enumerate(sorted(cluster_means.keys(), key=lambda x: cluster_means[x]))}
        return np.array([rename[i] for i in z])

    def calc_all_merge_costs(self):
        """
        Feed all possible pairs of clusters C(m, 2) to ClusterDict.calc_merge_cost
        This function is called only once, during initialization of Partition
        Subsequent calls operate on a subset of the data
        """
        pairs = itertools.combinations([k for k in self.clusters.keys()], 2)
        for pair in pairs:
            self.merge_costs[pair] = self.clusters.calc_merge_cost(*pair)

    def cluster_distance(self, i, j):
        """Euclidean distance between centroids of clusters i and j"""
        xi = np.mean([c.x for c in self.clusters[i]])
        xj = np.mean([c.x for c in self.clusters[j]])
        return np.linalg.norm(xi - xj)

    def find_merge_pair(self):
        """
        Search all cluster pairs for the best pair to merge.
        Use the following criteria:
        1) Find pair(s) for which merge cost is minimized
        2) If multiple candidates from (1), find pair with smallest inter-cluster distance
        """
        min_pair = min(self.merge_costs, key=lambda x: self.merge_costs[x])
        min_val = self.merge_costs[min_pair]
        assert min_val == self.clusters.calc_merge_cost(*min_pair)
        ties = [k for k, v in self.merge_costs.items() if v == min_val]
        if len(ties) > 1:
            d = {pair: self.cluster_distance(*pair) for pair in ties}
            min_pair = min(d, key=lambda x: d[x])
            assert(self.merge_costs[min_pair] == min_val)
        return min_pair

    def merge_next(self):
        """
        Iterate the AIB algorithm.
        Find best pair to merge, perform merge, update clusters and merge costs for next iteration
        """
        def _update(pair, new, old):
            if old in pair:
                pair = list(pair)
                pair[pair.index(old)] = new
                if pair[0] != pair[1]:  # Only return if this doesn't result in a self-pair
                    return tuple(pair)
            else:
                return pair

        min_pair = self.find_merge_pair()
        self.clusters.merge(*min_pair)
        """After merge, recompute costs related to the merged clusters
        Two steps:
            1) Update pointers to point to the merged pair (the min of min_pair)
            2) Process this list with clusters.calc_merge_cost
        """
        # First remove entries in merge_costs that are obsolete after merger
        remove = [key for key in self.merge_costs.keys()
                  if any([k in min_pair for k in key])]
        self.merge_costs = {key: value for key, value in self.merge_costs.items()
                            if key not in remove}
        # Now compute costs for updated pairs and add to merge_costs
        new, old = sorted(min_pair)
        new_pairs = [_update(pair, new, old) for pair in remove]
        # Eliminate duplicates using set comprehension
        # Eliminate "self pairs" which will show up in new_pairs list as None
        new_pairs = {pair for pair in new_pairs if pair}
        new_costs = {pair: self.clusters.calc_merge_cost(*pair) for pair in new_pairs}
        self.merge_costs.update(new_costs)


def preprocess(data, n_states):
    """If the data are approximately continuous, start by approximating the data by
    a fine-grained quantization into M (or fewer, depending on distribution!) distinct values"""
    _, bins = np.histogram(data, bins=n_states)
    index = np.digitize(data, np.concatenate((bins[:-1], [np.inf]))) - 1
    # ^ digitize handles rightmost bin differently than histogram
    centers = .5 * (bins[1:] + bins[:-1])  # bin centers are the quantized values for each state
    valmap = {i: val for i, val in enumerate(centers)}
    return np.fromiter((valmap[i] for i in index), dtype=float)


def aib(data, relevance_variable, n_init_states=None):
    """
    Run the Agglomerative Information Bottleneck algorithm (Slonim & Tishby, NIPS, 1999)

    :param data: the original data (X)
    :type data: numpy.ndarray

    :param relevance_variable: the relevance variable (Y)
    :type relevance_variable: numpy.ndarray

    :param n_init_states: If not none, start the algorithm from a crude discretization
        into this many linearly-spaced bins. Strongly recommended for large real-valued data.
    :type n_init_states: int (or float representation of integer)

    :return result: dict storing the discretization at each iteration
        e.g., result[3] will return a tuple specifying one of three values for each row in data

    :return score: dict storing the mutual information between the m-discretization and the relevance variable
        Can be used to select the appropriate value for m = |Z|
    """
    if len(data.shape) > 1 and np.prod(data.shape) > max(data.shape):
        raise NotImplementedError("Currently, aib is implemented only for 1-dimensional data. "
                                  "Multi-dimensional data can be discretized one variable at a time "
                                  "with respect to the relevance variable.")

    # Pre-process data with fine grid, M_init << N
    if n_init_states:
        data = preprocess(data, n_init_states)

    z = Partition(data, relevance_variable)
    result = {z.m: z.assignments}
    score = {z.m: mutual_information(z.assignments, relevance_variable)}
    while z.m > 1:
        z.merge_next()
        m = z.m
        d = np.array(z.assignments)
        result[m] = d
        score[m] = mutual_information(d, relevance_variable)
        print("Partition computed for |Z| = {}".format(m), flush=True)
    return result, score
