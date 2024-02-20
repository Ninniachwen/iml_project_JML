import inspect
import os
import sys
import unittest
import numpy as np

import torch

# access model in parent dir: https://stackoverflow.com/a/11158224/14934164
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.evaluation as e
import src.simplex_versions as s
import src.classifier_versions as c
import src.main as m
from original_code.src.simplexai.models.image_recognition import MnistClassifier
from src.Models.CatsAndDogsModel import CatsandDogsClassifier
from src.Models.HeartfailureModel import HeartFailureClassifier

class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print(10*"-" + "1. Unit tests" + 10*"-")
        self.random_seed = 42
        self.decomposition_size = 5
        self.corpus_size = 100
        self.test_size = 10
        self.test_id = 0
     
      
    def test_shuffle_data_loader(self):  # compare first element from loaded dataset
        print(3*">" + "1.1" + "testing shuffle data loader")

        # not shuffled, same first sample
        _, corpus1, test1 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
        _, corpus2, test2 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
        self.assertTrue(torch.equal(corpus1[0][0], corpus2[0][0]), "mnist corpus loader is shuffeling!")
        self.assertTrue(torch.equal(test1[0][0], test2[0][0]), "mnist test loader is shuffeling!")

        # one unshuffled, one shuffled, different first sample
        _, corpus3, test3 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
        self.assertFalse(torch.equal(corpus1[0][0], corpus3[0][0]), "mnist corpus loader is not shuffeling!")
        self.assertFalse(torch.equal(test1[0][0], test3[0][0]), "mnist test loader is not shuffeling!")

        # both shuffled, different first sample
        _, corpus4, test4 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
        self.assertFalse(torch.equal(corpus4[0][0], corpus3[0][0]), "mnist corpus loader is always shuffeling in the same way!")
        self.assertFalse(torch.equal(test4[0][0], test3[0][0]), "mnist test loader is not always shuffeling in the same way!")
    
    def test_r_2_scores(self):
        print(3*">" + "1.2" + "testing r2 scores")
        c1, corpus1, test1 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=True)
        c2, corpus2, test2 = c.train_or_load_mnist(42, 0, 100, 10, random_dataloader=True)
        result = e.r_2_scores(c1, corpus1[1], corpus1[1])
        self.assertEqual(result[0], 1.0, "score of original vs. original should be 1.0")

        result = e.r_2_scores(c1, corpus1[1], corpus2[1])
        self.assertNotEqual(result[0], 1.0)    # score of two different samples should not be 1.0
        
    def test_jacobians(self):
        print("TODO:  1.3 implement test_jacobians")# TODO: implement
        # test original jacobian method against ours

    def test_data_loaders(self):
        # check if all loaders return same format
        loaders = [c.train_or_load_mnist]#, c.train_or_load_heartfailure_model , c.train_or_load_CaN_model]
        types = [MnistClassifier, HeartFailureClassifier, CatsandDogsClassifier]
        for loader in loaders:
            result = loader(random_seed=self.random_seed, cv=0, corpus_size=self.corpus_size, test_size=self.test_size, random_dataloader=False)

            # base shape: triple
            self.assertEqual(len(result), 3, f"loader {loader} does not return a triple!")
            
            # shape/type of each triple item
            self.assertTrue(type(result[0]) in types, f"loader {loader} does not return one of the three classifier as first item!\n >> got {result[0]}, expected one of {types}")
            self.assertEqual(len(result[1]), 3, f"loader {loader} does not return a triple for corpusset!")
            self.assertEqual(len(result[2]), 3, f"loader {loader} does not return a triple for testset!")

            # corpus triple shapes & types
            self.assertTrue(
                (type(result[1][0]) == torch.Tensor) 
                & (len(result[1][0]) == self.corpus_size )
                & (result[1][0].dtype == torch.float32), 
                f"Corpus triple should contain a float32 tensor of length {self.corpus_size} as first item.\
                got {type(result[1][0])}, {len(result[1][0])}, {result[1][0].dtype}")
            self.assertTrue(
                (type(result[1][1]) == torch.Tensor) 
                & (list(result[1][1].shape) == [self.corpus_size] )
                & (result[1][1].dtype == torch.int64), 
                f"Corpus triple should contain a int64 tensor of shape [{self.corpus_size}] as second item.\
                got {type(result[1][1])}, {list(result[1][1].shape)}, {result[1][1].dtype}")
            self.assertTrue(
                (type(result[1][2]) == torch.Tensor) 
                & (len(result[1][2]) == self.corpus_size )
                & (result[1][2].dtype == torch.float32), 
                f"Corpus triple should contain a float32 tensor of length {self.corpus_size} as third item.\
                got {type(result[1][2])}, {len(result[1][2])}, {result[1][2].dtype}")

            # test triple shapes & types
            self.assertTrue(
                (type(result[2][0]) == torch.Tensor) 
                & (len(result[2][0]) == self.test_size )
                & (result[2][0].dtype == torch.float32), 
                f"Test triple should contain a float32 tensor of length {self.test_size} as first item.\
                got {type(result[2][0])}, {len(result[2][0])}, {result[2][0].dtype}")
            self.assertTrue(
                (type(result[2][1]) == torch.Tensor) 
                & (list(result[2][1].shape) == [self.test_size] )
                & (result[2][1].dtype == torch.int64), 
                f"Test triple should contain a int64 tensor of shape [{self.test_size}] as second item.\
                got {type(result[2][1])}, {list(result[2][1].shape)}, {result[2][1].dtype}")
            self.assertTrue(
                (type(result[2][2]) == torch.Tensor) 
                & (len(result[2][2]) == self.test_size )
                & (result[2][2].dtype == torch.float32), 
                f"Test triple should contain a float32 tensor of length {self.test_size} as third item.\
                got {type(result[2][2])}, {len(result[2][2])}, {result[2][2].dtype}")
            

    # edge cases für input var (testset > corpus)

    # exceptions
        
    #TODO: test random seed for classifier or simplex model?

    
class TestSimplex(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print(10*"-" + "0. training simplex" + 10*"-")
        
        models = [m.Model_Type.ORIGINAL, m.Model_Type.ORIGINAL_COMPACT, m.Model_Type.REIMPLEMENTED]     # without ablation models
        self.decomposition_size = 5
        self.corpus_size = 100
        self.test_size = 10
        self.test_id = 0
        self.results = []
        for model in models:
            w, lr2, or2, j, d = m.do_simplex(
                model_type=model, 
                decomposition_size=self.decomposition_size, 
                corpus_size = self.corpus_size,
                test_size = self.test_size,
                test_id = self.test_id,
                r_2_scores=True, 
                random_dataloader=False # we want the same sample set for each model to train on
                )  
            self.results.append({"w": w, "lr2": lr2, "or2": or2, "jac": j, "dec":d})
            # weights, latent_r2_score, output_r2_score, jacobian, decompostions

        self.sample_id = 0
        self.most_imp = 0

    def test_simplex_size(self):
        """
        check return shapes of do_simplex
        """
        print(10*"-" + "2. testing do_simplex" + 10*"-")
        print(3*">" + "2.1" + "testing do_simplex shapes & types")
        # weigts
        self.assertEqual(list(self.results[0]["w"].shape), [self.test_size, self.corpus_size], "weights of original model have incorrect shape!")
        self.assertEqual(list(self.results[1]["w"].shape), [self.test_size, self.corpus_size], "weights of compact model have incorrect shape!")
        self.assertEqual(list(self.results[2]["w"].shape), [self.test_size, self.corpus_size], "weights of reimplemented model have incorrect shape!")
        
        # latent_r2_score
        self.assertEqual(type(self.results[0]["lr2"]), np.float64, "latent_r2_score of original model have incorrect type!")
        self.assertEqual(type(self.results[1]["lr2"]), np.float64, "latent_r2_score of compact model have incorrect type!")
        self.assertEqual(type(self.results[2]["lr2"]), np.float64, "latent_r2_score of reimplemented model have incorrect type!")
        
        # output_r2_score
        self.assertEqual(type(self.results[0]["or2"]), np.float64, "output_r2_score of original model have incorrect shape!")
        self.assertEqual(type(self.results[1]["or2"]), np.float64, "output_r2_score of compact model have incorrect shape!")
        self.assertEqual(type(self.results[2]["or2"]), np.float64, "output_r2_score of reimplemented model have incorrect shape!")
        
        # jacobian
        self.assertEqual(self.results[0]["jac"].shape[0], self.corpus_size, "jacobian of original model have incorrect shape!")
        #self.assertEqual(self.results[1]["jac"].shape[0], self.corpus_size, "jacobian of compact model have incorrect shape!") #TODO: implenent jacobians
        self.assertEqual(self.results[2]["jac"].shape[0], self.corpus_size, "jacobian of reimplemented model have incorrect shape!")
        
        # decompostion shape
        orig_decomp = self.results[0]["dec"]
        compact_decomp = self.results[1]["dec"]
        rempl_decomp = self.results[2]["dec"]
        self.assertEqual(len(orig_decomp), self.test_size, "decomposition of original model have incorrect shape! (not a decomposition for each test sample)")
        self.assertEqual(len(compact_decomp), self.test_size, "decomposition of compact model have incorrect shape! (not a decomposition for each test sample)")
        self.assertEqual(len(rempl_decomp), self.test_size, "decomposition of reimplemented model have incorrect shape! (not a decomposition for each test sample)")

        # decomposition types & structure
        # {'sample_id': 0, 'img': tensor(), 'target': 7, 'decomposition': [
        #    {'c_id': 84, 'c_weight': 0.79606116, 'c_img': tensor(), 'c_target': 7}, ...
        # ]}
        self.assertEqual(type(rempl_decomp[0]["sample_id"]), int, "decomposition of reimplemented model have incorrect type! (sample_id no integer)")
        self.assertEqual(type(rempl_decomp[0]["img"]), torch.Tensor, "decomposition of reimplemented model have incorrect type! (sample img no tensor)")
        self.assertEqual(type(rempl_decomp[0]["target"]), int, "decomposition of reimplemented model have incorrect type! (target no integer)")
        self.assertEqual(type(rempl_decomp[0]["decomposition"][self.sample_id]["c_id"]), np.int64, "decomposition of reimplemented model have incorrect type! (corpus id no int)")
        self.assertEqual(type(rempl_decomp[0]["decomposition"][self.sample_id]["c_weight"]), np.float32, "decomposition of reimplemented model have incorrect type! (corpus weight no flaot)")
        self.assertEqual(type(rempl_decomp[0]["decomposition"][self.sample_id]["c_img"]), torch.Tensor, "decomposition of reimplemented model have incorrect type! (corpus img no tensor)")
        self.assertEqual(type(rempl_decomp[0]["decomposition"][self.sample_id]["c_target"]), int, "decomposition of reimplemented model have incorrect type! (corpus target no int)")



    def test_simplex_versions_decomposition(self):
        print(3*">" + "2.2" + "testing decomposition contents")
        # same sample?
        self.assertEqual(self.results[0]["dec"][self.sample_id]["sample_id"], self.results[1]["dec"][self.sample_id]["sample_id"], "got different samples!")
        self.assertEqual(self.results[0]["dec"][self.sample_id]["sample_id"], self.results[2]["dec"][self.sample_id]["sample_id"], "got different samples!")

        
        # descending order of corpus importance in decomposition (evaluation.py decompose)

        
        # quality tests:
        print(10*"-" + "testing quality of decompositions" + 10*"-")

        print(" " + 3*">" + "3.1" + "testing for most important explainer in decomp")
        # most important explainer is same class as sample (tested only for first sample, not all 10)?
        self.assertEqual(self.results[0]["dec"][self.sample_id]["target"], self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (Original Simplex)")
        self.assertEqual(self.results[1]["dec"][self.sample_id]["target"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (compact simplex)")
        self.assertEqual(self.results[2]["dec"][self.sample_id]["target"], self.results[2]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (reimplemented simplex)")
        
        print(" " + 3*">" + "3.2" + "comparing most important explainer in decomp between models")
        # check if explainer id's for first img is same for all models
        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & compact simplex!")
        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw compact & reimplemented simplex!")

        # check if same order of decomposition-id


        print(" " + 3*">" + "3.3" + "comparing importance of explainers between models")
        # same probability for all explainers (btw models) ?
        places_acccuracy = 4
        orig_top_weigts = [self.results[0]["dec"][self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        compact_top_weights = [self.results[1]["dec"][self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        reimpl_top_weights = [self.results[1]["dec"][self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        
        self.assertAlmostEqual(orig_top_weigts, compact_top_weights, places=places_acccuracy, msg="accuracy btw original & compact simplex differs!")
        self.assertAlmostEqual(orig_top_weigts, reimpl_top_weights, places=places_acccuracy, msg="accuracy btw compact & reimplemented simplex differs!")

        


        
    # decomposition needs to add up to ~100% 
        
        
    # maybe test class-distr of classification against class-distr of decomposition
        
    # does test_size influence simplex performance?

if __name__ == "__main__":
    #unittest.main()
    test = UnitTests()
    test.setUpClass()
    test.test_data_loaders()


# execute all tests via console from root dir using
# python -m tests.unittests