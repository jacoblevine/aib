__author__ = 'jacoblevine'
from sklearn.datasets import load_iris
import unittest


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.data = load_iris().data[:, 2]  # this variable looks reasonably discrete, relative to target
        self.relevance_variable = load_iris().target

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
