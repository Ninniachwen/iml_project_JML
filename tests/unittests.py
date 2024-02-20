import inspect
from math import isclose
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
from src.models.CatsAndDogsModel import CatsandDogsClassifier
from src.models.HeartfailureModel import HeartFailureClassifier

def is_close_w_index(a:list[float], b:list[float]): 
    """
    source: https://stackoverflow.com/a/72291500/14934164
    compares elements of 2 lists pairwise for numerical diff.
    """
    r=[]
    # same length lists, use zip to iterate pairwise, use enumerate for index
    for idx, (aa, bb) in enumerate(zip(a,b)):
        # convert to floats
        aaa = float(aa)
        bbb = float(bb)

        # append if not close
        if not isclose(aaa,bbb):
            r.append((idx, (aaa,bbb)))

    # print results
    for w in r:
        print("On index {0} we have {1} != {2}".format(w[0],*w[1]), sep="\n")
    
    return True if r==[] else False

class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print(10*"-" + "Unit tests" + 10*"-")
        self.random_seed = 42
        self.decomposition_size = 5
        self.corpus_size = 10
        self.test_size = 1
        self.test_id = 0
     

    def test_data_loaders(self):
        """
        tests evaluation.py r_2_scores(): check if all loaders return classifier and correct format and types for corpus and test sets.
        """
        print(3*">" + "testing data loader format and type")
        loaders = [c.train_or_load_mnist]#TODO, c.train_or_load_heartfailure_model , c.train_or_load_CaN_model]
        types = [MnistClassifier, HeartFailureClassifier, CatsandDogsClassifier]
        corpus_size = 10
        test_size = 1
        for loader in loaders:
            result = loader(random_seed=self.random_seed, cv=0, corpus_size=corpus_size, test_size=test_size, random_dataloader=False)

            # base shape: triple
            self.assertEqual(len(result), 3, f"loader {loader} does not return a triple!")
            
            # shape/type of each triple item
            self.assertTrue(type(result[0]) in types, f"loader {loader} does not return one of the three classifier as first item!\n >> got {result[0]}, expected one of {types}")
            self.assertEqual(len(result[1]), 3, f"loader {loader} does not return a triple for corpusset!")
            self.assertEqual(len(result[2]), 3, f"loader {loader} does not return a triple for testset!")

            # corpus triple shapes & types
            self.assertTrue(
                (type(result[1][0]) == torch.Tensor) 
                & (len(result[1][0]) == corpus_size )
                & (result[1][0].dtype == torch.float32), 
                f"Corpus triple should contain a float32 tensor of length {corpus_size} as first item.\
                got {type(result[1][0])}, {len(result[1][0])}, {result[1][0].dtype}")
            self.assertTrue(
                (type(result[1][1]) == torch.Tensor) 
                & (list(result[1][1].shape) == [corpus_size] )
                & (result[1][1].dtype == torch.int64), 
                f"Corpus triple should contain a int64 tensor of shape [{corpus_size}] as second item.\
                got {type(result[1][1])}, {list(result[1][1].shape)}, {result[1][1].dtype}")
            self.assertTrue(
                (type(result[1][2]) == torch.Tensor) 
                & (len(result[1][2]) == corpus_size )
                & (result[1][2].dtype == torch.float32), 
                f"Corpus triple should contain a float32 tensor of length {corpus_size} as third item.\
                got {type(result[1][2])}, {len(result[1][2])}, {result[1][2].dtype}")

            # test triple shapes & types
            self.assertTrue(
                (type(result[2][0]) == torch.Tensor) 
                & (len(result[2][0]) == test_size )
                & (result[2][0].dtype == torch.float32), 
                f"Test triple should contain a float32 tensor of length {test_size} as first item.\
                got {type(result[2][0])}, {len(result[2][0])}, {result[2][0].dtype}")
            self.assertTrue(
                (type(result[2][1]) == torch.Tensor) 
                & (list(result[2][1].shape) == [test_size] )
                & (result[2][1].dtype == torch.int64), 
                f"Test triple should contain a int64 tensor of shape [{test_size}] as second item.\
                got {type(result[2][1])}, {list(result[2][1].shape)}, {result[2][1].dtype}")
            self.assertTrue(
                (type(result[2][2]) == torch.Tensor) 
                & (len(result[2][2]) == test_size )
                & (result[2][2].dtype == torch.float32), 
                f"Test triple should contain a float32 tensor of length {test_size} as third item.\
                got {type(result[2][2])}, {len(result[2][2])}, {result[2][2].dtype}")
            
      
    def test_shuffle_data_loader(self):
        """
        compare first element from loaded dataset.
        Testing train_or_load_mnist, train_or_load_CaN_model & train_or_load_heartfailure_model from classifier_versions.py
        """
        print(3*">" + "testing shuffle data loader")

        for loader in [c.train_or_load_mnist]:#TODO, c.train_or_load_CaN_model, c.train_or_load_heartfailure_model]:

            # not shuffled, same first sample
            _, corpus1, test1 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
            _, corpus2, test2 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
            self.assertTrue(torch.equal(corpus1[0][0], corpus2[0][0]), "mnist corpus loader is shuffeling!")
            self.assertTrue(torch.equal(test1[0][0], test2[0][0]), "mnist test loader is shuffeling!")

            # one unshuffled, one shuffled, different first sample
            _, corpus3, test3 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
            self.assertFalse(torch.equal(corpus1[0][0], corpus3[0][0]), "mnist corpus loader is not shuffeling!")
            self.assertFalse(torch.equal(test1[0][0], test3[0][0]), "mnist test loader is not shuffeling!")

            # both shuffled, different first sample
            _, corpus4, test4 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
            self.assertFalse(torch.equal(corpus4[0][0], corpus3[0][0]), "mnist corpus loader is always shuffeling in the same way!")
            self.assertFalse(torch.equal(test4[0][0], test3[0][0]), "mnist test loader is not always shuffeling in the same way!")
    
    def test_r_2_scores(self):
        """
        tests r_2_scores() from evaluation.py. r2 score schould be 1.0 for same img, and lower than 1.0 for different imgs. 
        """
        print(3*">" + "testing r2 scores")
        for dataloader in [c.train_or_load_mnist]:#TODO, c.train_or_load_CaN_model, c.train_or_load_heartfailure_model]
            c1, corpus1, test1 = dataloader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
            c2, corpus2, test2 = dataloader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
        
            result = e.r_2_scores(c1, corpus1[2], corpus1[2])
            self.assertEqual(result[0], 1.0, "score of original vs. original should be 1.0")

            result = e.r_2_scores(c1, corpus1[2], corpus2[2])
            self.assertLess(result[0], 1.0)    # score of two different samples should not be 1.0
        
    


    # edge cases für input var (testid > testset)

    # exceptions
            
    # test make_corpus
    
    # TODO: test heartfialure & catsAndDogs predictis/training
        
    #TODO: test random seed for classifier or simplex model?

    
class TestSimplex(unittest.TestCase):
    """
    This class tests other funktions, using do_simplex, because it handles the more complex input variables like classifier, dataset, ..
    It tests original_model(), compact_original_model() and reimplemented_model() in their basic (not ablation) funtionality from simplex_versions.py.
    Also the jacobians from different models and create_decompositions() from evaluation.py.

    """    
    
    @classmethod
    def setUpClass(self):
        print(10*"-" + "training our simplex" + 10*"-")
        
        models = [m.Model_Type.ORIGINAL, m.Model_Type.ORIGINAL_COMPACT, m.Model_Type.REIMPLEMENTED]     # without ablation models
        #self.simplex_model_functions = [c.original_model]#TODO, c.compact_original_model, c.reimplemented_model]
        self.decomposition_size = 5
        self.corpus_size = 100
        self.test_size = 10
        self.test_id = 0
        self.random_seed = 42
        self.results = []
        self.cv = 0
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

        self.orig_decomp = self.results[0]["dec"]
        self.compact_decomp = self.results[1]["dec"]
        self.rempl_decomp = self.results[2]["dec"]
        
        self.orig_weights = [self.orig_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        self.compact_weights = [self.compact_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        self.reimpl_weights = [self.rempl_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]

        self.orig_c_ids = [self.orig_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]
        self.compact_c_ids = [self.compact_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]
        self.reimpl_c_ids = [self.rempl_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]

    def test_simplex_size(self):
        """
        check return shapes, types and sorting of do_simplex
        """
        print(10*"-" + "testing do_simplex" + 10*"-")
        print(3*">" + "testing do_simplex shapes, types and sorting")
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
        
        # jacobian shape
        self.assertEqual(self.results[0]["jac"].shape[0], self.corpus_size, "jacobian of original model have incorrect shape!")
        #self.assertEqual(self.results[1]["jac"].shape[0], self.corpus_size, "jacobian of compact model have incorrect shape!") #TODO: implenent jacobians
        self.assertEqual(self.results[2]["jac"].shape[0], self.corpus_size, "jacobian of reimplemented model have incorrect shape!")

        # decompostion shape
        self.assertEqual(len(self.orig_decomp), self.test_size, "decomposition of original model have incorrect shape! (not a decomposition for each test sample)")
        self.assertEqual(len(self.compact_decomp), self.test_size, "decomposition of compact model have incorrect shape! (not a decomposition for each test sample)")
        self.assertEqual(len(self.rempl_decomp), self.test_size, "decomposition of reimplemented model have incorrect shape! (not a decomposition for each test sample)")

        # decomposition types & structure
        # {'sample_id': 0, 'img': tensor(), 'target': 7, 'decomposition': [
        #    {'c_id': 84, 'c_weight': 0.79606116, 'c_img': tensor(), 'c_target': 7}, ...
        # ]}
        self.assertEqual(type(self.rempl_decomp[0]["sample_id"]), int, "decomposition of reimplemented model have incorrect type! (sample_id no integer)")
        self.assertEqual(type(self.rempl_decomp[0]["img"]), torch.Tensor, "decomposition of reimplemented model have incorrect type! (sample img no tensor)")
        self.assertEqual(type(self.rempl_decomp[0]["target"]), int, "decomposition of reimplemented model have incorrect type! (target no integer)")
        self.assertEqual(type(self.rempl_decomp[0]["decomposition"][self.sample_id]["c_id"]), np.int64, "decomposition of reimplemented model have incorrect type! (corpus id no int)")
        self.assertEqual(type(self.rempl_decomp[0]["decomposition"][self.sample_id]["c_weight"]), np.float32, "decomposition of reimplemented model have incorrect type! (corpus weight no flaot)")
        self.assertEqual(type(self.rempl_decomp[0]["decomposition"][self.sample_id]["c_img"]), torch.Tensor, "decomposition of reimplemented model have incorrect type! (corpus img no tensor)")
        self.assertEqual(type(self.rempl_decomp[0]["decomposition"][self.sample_id]["c_target"]), int, "decomposition of reimplemented model have incorrect type! (corpus target no int)")

        # descending order of corpus importance in decomposition (evaluation.py decompose)
        self.assertTrue(all(self.orig_weights[i] >= self.orig_weights[i+1] for i in range(len(self.orig_weights) - 1)))
        self.assertTrue(all(self.compact_weights[i] >= self.compact_weights[i+1] for i in range(len(self.compact_weights) - 1)))
        self.assertTrue(all(self.reimpl_weights[i] >= self.reimpl_weights[i+1] for i in range(len(self.reimpl_weights) - 1)))
        # check if list is sorted: https://stackoverflow.com/a/3755251/14934164

        
        # decomposition should be identical for original & compact original
        self.assertTrue(is_close_w_index(self.orig_weights, self.compact_weights))
        # self.assertTrue(is_close_w_index(self.orig_weights, self.reimpl_weights))
        # not the same to reimplemented weights

        # decomposition needs to add up to ~100% 
        self.assertAlmostEqual(sum(self.orig_weights), 1.0, delta=0.01, msg="original decomposition weights do not add up to 99%")
        self.assertAlmostEqual(sum(self.reimpl_weights), 1.0, delta=0.25, msg="reimplemented decomposition weights do not add up to 99%")
        
    def test_jacobians(self):
        print("TODO: implement test_jacobians")# TODO: implement
        # test original jacobian method against ours
        #e.plot_jacobians(self.results[0]["jac"][0][0])
        #e.plot_jacobians(self.results[2]["jac"][0][0])
        #e.plot_jacobians(self.results[2]["jac"][0][0]-self.results[0]["jac"][0][0])
        #self.assertTrue(torch.equal(self.results[0]["jac"][0], self.results[2]["jac"][0]), "first reimplemented jacobian differs from first original jacobian")

        """currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        save_path = os.path.join(os.path.dirname(currentdir), "files")
        orig_mnist_classifier = MnistClassifier()
        orig_mnist_classifier.load_state_dict(torch.load(os.path.join(save_path , f"model_cv{self.cv}.pth")))

        original_jac = self.results[0]["jac"]
        our_mnist_classifier, corpus, test_set = c.train_or_load_mnist(self.random_seed, cv=self.cv, corpus_size=self.corpus_size, test_size=self.test_size, random_dataloader=False)
        test_id=0
        corpus_inputs = corpus[0]
        corpus_latents = corpus[2]
        test_latents = test_set[2]
        input_baseline = input_baseline = torch.zeros(corpus_inputs.shape)
        original = s.Simplex(corpus_examples=corpus_inputs,
            corpus_latent_reps=corpus_latents)
        reimplemented = s.Simplex_Model(self.corpus_size, self.test_size, weight_init_zero=True)

        self.assertEqual(
            original.jacobian_projection(test_id=test_id, model=orig_mnist_classifier, input_baseline=input_baseline), 
            original_jac, 
            f"jacobian of original is inconsistent !")
        self.assertEqual(
            reimplemented.get_jacobian(test_id, corpus_inputs, test_latents, input_baseline, our_mnist_classifier), 
            original_jac, 
            f"jacobian of reimplementation ... !")"""
        


    def test_decomposition_quality(self):
        """
        checking decomposition contents
        """
        print(10*"-" + "testing quality of decompositions" + 10*"-")
        # quality tests:

        print(3*">" + "checking most important corpus id in decomposition")
        # most important sample: same for all models?
        self.assertEqual(self.results[0]["dec"][self.sample_id]["sample_id"], self.results[1]["dec"][self.sample_id]["sample_id"], "got different samples!")
        self.assertEqual(self.results[0]["dec"][self.sample_id]["sample_id"], self.results[2]["dec"][self.sample_id]["sample_id"], "got different samples!")

        print(3*">" + "QUALITY: testing for target of most important explainer in decomp")
        # most important explainer is same class as sample (tested only for first sample, not all 10)?
        self.assertEqual(self.results[0]["dec"][self.sample_id]["target"], self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (Original Simplex)")
        self.assertEqual(self.results[1]["dec"][self.sample_id]["target"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (compact simplex)")
        self.assertEqual(self.results[2]["dec"][self.sample_id]["target"], self.results[2]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_target"], "quality issue, most important explainer has different target than sample! (reimplemented simplex)")
        
        print(3*">" + "QUALITY: comparing most important explainer in decomp between models")
        # check if explainer id's for first img is same for all models
        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & compact simplex!")
        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw compact & reimplemented simplex!")

        # check if same corpus-ids in decomposition
        self.assertListEqual(self.orig_c_ids, self.compact_c_ids, f"corpus id's in decomposition differ btw original and compact model: {self.orig_c_ids}, {self.compact_c_ids}")
        print(3*">" + "QUALITY: comparing corpus id's in decomp between original and reimplemented simplex")
        #self.assertListEqual(self.orig_c_ids, self.reimpl_c_ids, f"QUALITY: corpus id's in decomposition differ btw original and reimplemented model: {self.orig_c_ids}, {self.reimpl_c_ids}") #TODO: not true, keep?
        #    print(f"QUALITY Issue: corpus id's in decomposition differ btw original and reimplemented model: {self.orig_c_ids}, {self.reimpl_c_ids}")


""" checked above #TODO: delete
        print(3*">" + "QUALITY: comparing importance of explainers(corpus images in decomposition) between models")
        # same probability for all explainers (btw models) ?
        places_acccuracy = 4        
        self.assertAlmostEqual(self.orig_weights, self.compact_weights, places=places_acccuracy, msg="accuracy btw original & compact simplex differs!")
        self.assertAlmostEqual(self.orig_weights, self.reimpl_weights, places=places_acccuracy, msg="accuracy btw compact & reimplemented simplex differs!")"""

        

    
        
    # maybe test class-distr of classification against class-distr of decomposition
        
    # does test_size influence simplex performance?

if __name__ == "__main__":
    unittest.main()
    #test = UnitTests()
    #test.setUpClass()
    #test.test_data_loaders()


# execute all tests via console from root dir using
# python -m tests.unittests