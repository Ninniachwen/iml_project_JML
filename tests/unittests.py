import inspect
import os
import sys
import unittest

import torch

# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.evaluation as e
import src.simplex_versions as s
import src.classifier_versions as c

#sys.path.insert(0, "")
#WORKING_DIR = "src"
#module = __import__(f"{WORKING_DIR}.classifier_versions", fromlist=['train_or_load_mnist'])

class TestAll(unittest.TestCase):

    def test_seeded_data_loader(self):  # compare first element from loaded dataset
        _, corpus1, test1 = c.train_or_load_mnist(42, 0, 100, 10)
        _, corpus2, test2 = c.train_or_load_mnist(42, 0, 100, 10)
        self.assertTrue(torch.equal(corpus1[0][0], corpus2[0][0]), "mnist corpus load is not seeded!")
        self.assertTrue(torch.equal(test1[0][0], test2[0][0]), "mnist test load is not seeded!")

        # different seed, unequal corpus and test set
        _, corpus3, test3 = c.train_or_load_mnist(43, 0, 100, 10)
        self.assertFalse(torch.equal(corpus1[0][0], corpus3[0][0]), "mnist corpus load is not seeded, should be different for different seeds")
        self.assertFalse(torch.equal(test1[0][0], test3[0][0]), "mnist test load is not seeded, should be different for different seeds")

    def test_r_2_scores(self):
        c1, corpus1, test1 = c.train_or_load_mnist(42, 0, 100, 10)
        c2, corpus2, test2 = c.train_or_load_mnist(43, 0, 100, 10)
        result = e.r_2_scores(c1, corpus1[1], corpus1[1])
        self.assertEqual(result[0], 1.0, "score of original vs. original should be 1.0")

        result = e.r_2_scores(c1, corpus1[1], corpus2[1])
        self.assertNotEqual(result[0], 1.0)    # score of two very different samples should be 1.0
        
    def test_jacobians():
        print("test")

if __name__ == "__main__":
    test = TestAll()
    test.test_seeded_data_loader()
    test.test_r_2_scores()
