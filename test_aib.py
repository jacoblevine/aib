__author__ = 'jacoblevine'
from sklearn.datasets import load_iris
import aib
import unittest


class IrisTestCase(unittest.TestCase):

    def setUp(self):
        self.data = load_iris().data[:, 2]  # this variable looks reasonably discrete, relative to target
        self.relevance_variable = load_iris().target

    def test_preprocess(self):
        m = 10
        data = aib.preprocess(self.data, m)
        self.assertEqual(len(set(data)), m)


if __name__ == '__main__':
    unittest.main()
