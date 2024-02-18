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
import src.main as m

class TestAll(unittest.TestCase):

    def test_shuffle_data_loader(self):  # compare first element from loaded dataset
        # not shuffled, same first sample
        _, corpus1, test1 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=False)
        _, corpus2, test2 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=False)
        self.assertTrue(torch.equal(corpus1[0][0], corpus2[0][0]), "mnist corpus loader is shuffeling!")
        self.assertTrue(torch.equal(test1[0][0], test2[0][0]), "mnist test loader is shuffeling!")

        # one unshuffled, one shuffled, different first sample
        _, corpus3, test3 = c.train_or_load_mnist(43, 0, 100, 10, random_dataloader=True)
        self.assertFalse(torch.equal(corpus1[0][0], corpus3[0][0]), "mnist corpus loader is not shuffeling!")
        self.assertFalse(torch.equal(test1[0][0], test3[0][0]), "mnist test loader is not shuffeling!")

        # both shuffled, different first sample
        _, corpus4, test4 = c.train_or_load_mnist(43, 0, 100, 10, random_dataloader=True)
        self.assertFalse(torch.equal(corpus4[0][0], corpus3[0][0]), "mnist corpus loader is always shuffeling in the same way!")
        self.assertFalse(torch.equal(test4[0][0], test3[0][0]), "mnist test loader is not always shuffeling in the same way!")
    
    def test_r_2_scores(self):
        c1, corpus1, test1 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=True)
        c2, corpus2, test2 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=True)
        result = e.r_2_scores(c1, corpus1[1], corpus1[1])
        self.assertEqual(result[0], 1.0, "score of original vs. original should be 1.0")

        result = e.r_2_scores(c1, corpus1[1], corpus2[1])
        self.assertNotEqual(result[0], 1.0)    # score of two very different samples should be 1.0
        
    def test_jacobians(self):
        print("TODO: implement test_jacobians")# TODO: implement
        # test original jacobian method against ours

    # check return shapes of do simplex
        
    # check if both loaders return same format
        

    # edge cases für input var (testset > corpus)

    # exceptions
        
    #TODO: test random seed for classifier or simplex model?
    

    def test_simplex_versions_decomposition(self): # without ablation
        models = [m.Model_Type.ORIGINAL, m.Model_Type.ORIGINAL_COMPACT, m.Model_Type.REIMPLEMENTED]
        decomposition_size = 5
        results = []
        for model in models:
            _, _, _, _, decomp = m.do_simplex(model_type=model, decomposition_size=decomposition_size, r_2_scores=False, random_dataloader=False)  # we want the same sample set for each model to train on
            results.append(decomp)
        
        sample_id = 0
        most_imp = 0
        
        # same sample?
        self.assertEqual(results[0][sample_id]["sample_id"], results[1][sample_id]["sample_id"], "got different samples!")
        self.assertEqual(results[0][sample_id]["sample_id"], results[2][sample_id]["sample_id"], "got different samples!")

        
        # descending order of corpus importance in decomposition (evaluation.py decompose)

        
        # quality tests:

        # most important explainer is same class as sample (tested only for first sample, not all 10)?
        self.assertEqual(results[0][sample_id]["target"], results[0][sample_id]["decomposition"][most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (Original Simplex)")
        self.assertEqual(results[1][sample_id]["target"], results[1][sample_id]["decomposition"][most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (compact simplex)")
        self.assertEqual(results[2][sample_id]["target"], results[2][sample_id]["decomposition"][most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (reimplemented simplex)")
        
        # check if explainer id's for first img is same for all models
        self.assertEqual(results[0][sample_id]["decomposition"][most_imp]["c_id"], results[1][sample_id]["decomposition"][most_imp]["c_id"], msg="most important explainer differs btw original & compact simplex!")
        self.assertEqual(results[0][sample_id]["decomposition"][most_imp]["c_id"], results[1][sample_id]["decomposition"][most_imp]["c_id"], msg="most important explainer differs btw compact & reimplemented simplex!")

        # check if same order of decomposition-id


        # same probability for all explainers (btw models) ?
        places_acccuracy = 4
        orig_top_weigts = [results[0][sample_id]["decomposition"][i]["c_weight"] for i in range(decomposition_size)]
        compact_top_weights = [results[1][sample_id]["decomposition"][i]["c_weight"] for i in range(decomposition_size)]
        reimpl_top_weights = [results[1][sample_id]["decomposition"][i]["c_weight"] for i in range(decomposition_size)]
        
        self.assertAlmostEqual(orig_top_weigts, compact_top_weights, places=places_acccuracy, msg="accuracy btw original & compact simplex differs!")
        self.assertAlmostEqual(orig_top_weigts, reimpl_top_weights, places=places_acccuracy, msg="accuracy btw compact & reimplemented simplex differs!")

        


        
    # decomposition needs to add up to ~100% 
        
        
    # maybe test class-distr of classification against class-distr of decomposition
        
    # does test_size influence simplex performance?

if __name__ == "__main__":
    unittest.main()


# execute all tests via console from root dir using
# python -m tests.unittests