import inspect
import math
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader
import unittest

sys.path.insert(0, "")

from original_code.src.simplexai.models.image_recognition import MnistClassifier
from original_code.src.simplexai.experiments import mnist
import src.classifier_versions as c
import src.evaluation as e
import src.main as m
import src.simplex_versions as s
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.classifier.HeartfailureClassifier import HeartFailureClassifier
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.datasets.heartfailure_dataset import HeartFailureDataset
from src.heartfailure_prediction import load_data
from src.utils.image_finder_cats_and_dogs import LABEL, get_images
from src.utils.corpus_creator import make_corpus
from src.utils.utlis import is_close_w_index, jacobian_compare_score, plot_jacobians_grayscale, plot_mnist, print_jacobians_with_img

class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print(10*"-" + "Unit tests" + 10*"-")
        self.random_seed = 42
        self.decomposition_size = 5
        self.corpus_size = 10
        self.test_size = 1
        self.test_id = 0
        self.loaders = [c.train_or_load_mnist, c.train_or_load_heartfailure_model , c.train_or_load_CaD_model]


    def test_get_images(self):
        """
        check separiatoin of corpus image.
        """
        image_paths = ["data/Animal Images/train", "data/Animal Images/test"]
        for image_path in image_paths:
            picture_files, labels = get_images(image_path)
            self.assertTrue(
                (len(picture_files)==len(labels))
                & all(label in LABEL.values() for label in labels))
            if image_path == image_path[0]:
                self.assertTrue(
                    (len(picture_files)==1000)# 1000 pictures in the test set
                    &(len(labels.count(LABEL["cats"])==labels.count(LABEL["dogs"])))# test set is balanced
                )
            if image_path == image_path[1]:
                self.assertTrue(
                    (len(picture_files)==29062)# 29062 pictures in the training set
                    &(len(labels.count(LABEL["cats"])==14560)) 
                    &(len(labels.count(LABEL["dogs"])==14502))
                )
     

    def test_data_loaders(self):
        """
        tests train_or_load methods from classifier_versions.py. check if all loaders return classifier and correct format and types for corpus and test sets.
        """
        print(3*">" + "testing data loader format and type")
        types = [MnistClassifier, HeartFailureClassifier, CatsandDogsClassifier]
        corpus_size = 10
        test_size = 1
        for loader in self.loaders:
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
                f"Corpus triple should contain a float32 tensor of length {corpus_size} as first item (data).\
                got {type(result[1][0])}, {len(result[1][0])}, {result[1][0].dtype}")
            self.assertTrue(
                (type(result[1][1]) == torch.Tensor) 
                & (list(result[1][1].shape) == [corpus_size] )
                & (result[1][1].dtype == torch.int64), 
                f"Corpus triple should contain a int64 tensor of shape [{corpus_size}] as second item (target).\
                got {type(result[1][1])}, {list(result[1][1].shape)}, {result[1][1].dtype}")
            self.assertTrue(
                (type(result[1][2]) == torch.Tensor) 
                & (len(result[1][2]) == corpus_size )
                & (result[1][2].dtype == torch.float32), 
                f"Corpus triple should contain a float32 tensor of length {corpus_size} as third item (latents).\
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
        Testing train_or_load_mnist, train_or_load_CaD_model & train_or_load_heartfailure_model from classifier_versions.py
        """
        print(3*">" + "testing shuffle data loader")
        for loader in self.loaders:

            # not shuffled, same first sample
            _, corpus1, test1 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
            _, corpus2, test2 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=False)
            self.assertTrue(torch.equal(corpus1[0][0], corpus2[0][0]), f"{loader} corpus loader is shuffeling!")
            self.assertTrue(torch.equal(test1[0][0], test2[0][0]), f"{loader} test loader is shuffeling!")

            # one unshuffled, one shuffled, different first sample
            _, corpus3, test3 = loader(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
            self.assertFalse(torch.equal(corpus1[0][0], corpus3[0][0]), f"{loader} corpus loader is not shuffeling!")
            self.assertFalse(torch.equal(test1[0][0], test3[0][0]), f"{loader} test loader is not shuffeling!")

            # both shuffled, different first sample
            _, corpus4, test4 = loader(self.random_seed+1, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
            self.assertFalse(torch.equal(corpus4[0][0], corpus3[0][0]), f"{loader} corpus loader is always shuffeling in the same way!")
            self.assertFalse(torch.equal(test4[0][0], test3[0][0]), f"{loader} test loader is not always shuffeling in the same way!")
    
    def test_make_corpus(self):
        """
        test make_corpus class distribution
        """
        image_paths, labels = get_images("data/Animal Images/test")
        dataset = CandDDataSet(image_paths=image_paths, labels=labels)
        datalaoder = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
        corpus = make_corpus(datalaoder, corpus_size=100)
        count = [list(corpus[1]).count(i) for i in [0,1]]
        self.assertEqual(count, [50,50], "corpus does not have a 50/50 class distribution")

        datapath = r"data\heart.csv"
        x,y = load_data(datapath)
        dataset = HeartFailureDataset(x, y)
        datalaoder = DataLoader(dataset, batch_size=100)
        corpus = make_corpus(datalaoder, corpus_size=100)
        count = [list(corpus[1]).count(i) for i in [0,1]]
        self.assertEqual(count, [50,50], "corpus does not have a 50/50 class distribution")
        
        datalaoder = mnist.load_mnist(batch_size=100, train=False)
        corpus = make_corpus(datalaoder, corpus_size=100, n_classes=10)
        classes = torch.unique(corpus[1])
        count = [list(corpus[1]).count(i) for i in classes]
        self.assertEqual(count, [100/len(classes) for i in classes], "corpus does not have a 50/50 class distribution")
        
    def test_r_2_scores(self):
        """
        tests r_2_scores() from evaluation.py. r2 score schould be 1.0 for same tensors, and lower than 1.0 for different tensors. 
        """
        print(3*">" + "testing r2 scores")
        c1, corpus1, test1 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
        c2, corpus2, test2 = c.train_or_load_mnist(self.random_seed, self.test_id, self.corpus_size, self.test_size, random_dataloader=True)
    
        result = e.r_2_scores(c1, corpus1[2], corpus1[2])
        self.assertEqual(result[0], 1.0, "score of identical tensors should be 1.0")

        result = e.r_2_scores(c1, corpus1[2], corpus2[2])
        self.assertLess(result[0], 1.0, "score of two different samples should be less than 1.0")

    def test_exceptions(self):
        """
        test exception for wrong datset or None result dataset-model combination
        """

        # exception datset
        with self.assertRaises(Exception) as context:
            m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset="NON_EXISTING_DATASET",
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False, #this is only a print toggle, the jacobians will still be created
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )
        self.assertTrue("no valid input for dataset" in str(context.exception), "do_simplex should raise an exception if an unknown dataset is given")
        
        # wrong model-dataset-combination
        self.assertEqual(m.do_simplex(
                    model_type=m.Model_Type.R_NORMALIZE_IWR,
                    dataset=m.Dataset.CaD,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False, #this is only a print toggle, the jacobians will still be created
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                ),None , "do_simplex should return None if an invald dataset-model combination is given")

        # exception (decompsition > corpus)
        with self.assertRaises(Exception) as context:
            m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset=m.Dataset.MNIST,
                    cv=0,
                    decomposition_size=10,
                    corpus_size=5,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False, #this is only a print toggle, the jacobians will still be created
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )
        self.assertTrue("decomposition size can't be larger than corpus" in str(context.exception), "do_simplex should raise an exception if decomposition size is larger than corpus")

        # exception (test_id > test_size)
        with self.assertRaises(Exception) as context:
            m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset=m.Dataset.MNIST,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=2,
                    print_jacobians=False, #this is only a print toggle, the jacobians will still be created
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )
        self.assertTrue("test_id can't be larger than test_size" in str(context.exception), "do_simplex should raise an exception if test_id is larger than test_size")

            
    def test_do_simplex(self):
        """
        testing return values of do_simplex for different edge cases
        """
        # if 4x false, result should be tuple[torch.Tensor, None, None, None, None]
        simplex_all_false = m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset=m.Dataset.MNIST,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False, #this is only a print toggle, the jacobians will still be created
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )
        self.assertTrue(
            (type(simplex_all_false[0])==type(simplex_all_false[3])==torch.Tensor)
            & (simplex_all_false[1]==simplex_all_false[2]==simplex_all_false[4]==None), 
            f"do_simplex should only return 2 tensors and 3 Nones in this setting. got {simplex_all_false}")
        
        # if test_size<3 then no r2 scores are created (restriciton from original simplex)
        simplex_NaN = m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset=m.Dataset.MNIST,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False,
                    r_2_scores=True,
                    decompose=True,
                    random_dataloader=False
                )
        self.assertTrue(
            (type(simplex_NaN[0])==type(simplex_NaN[3])==torch.Tensor)
            & math.isnan(simplex_NaN[1]) & (math.isnan(simplex_NaN[2]))
            & (type(simplex_NaN[4])==list),
            f"do_simplex should only return NaNs for r2 scores in this setting. got {simplex_NaN[1]} & {simplex_NaN[2]}")
        
        # return weights, latent_r2_score, output_r2_score, jacobian, decompostions
        # tuple[torch.Tensor, list[float], list[float], torch.Tensor, list[dict]]
        # for mnist, all model types should give valid return values
        for mod in m.Model_Type:
            result = m.do_simplex(
                model_type=mod,
                dataset=m.Dataset.MNIST,
                cv=0,
                decomposition_size=3,
                corpus_size=10,
                test_size=3,
                test_id=0,
                print_jacobians=False,
                r_2_scores=True,
                decompose=True,
                random_dataloader=True
            )
            self.assertTrue(
                (type(result[0])==type(result[3])==torch.Tensor) 
                & (type(result[1])==type(result[2])==np.float64)
                & (type(result[4])==list), 
                f"do_simplex should only return weights tensor and Nones in this setting ({mod}). got {result}")
                
        # return weights, latent_r2_score, output_r2_score, jacobian, decompostions
        # tuple[torch.Tensor, list[float], list[float], torch.Tensor, list[dict]]
        # for other Datasets, the first three (non Ablation) model types should give valid return values
        for d in list(m.Dataset)[1:]: #because [0]->MNIST is already tested above
            for mod in list(m.Model_Type)[:3]:
                result = m.do_simplex(
                    model_type=mod,
                    dataset=d,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=3,
                    test_id=0,
                    print_jacobians=False,
                    r_2_scores=True,
                    decompose=True,
                    random_dataloader=False
                )
                self.assertTrue(
                    (type(result[0])==type(result[3])==torch.Tensor)
                    & (type(result[1])==type(result[2])==np.float64)
                    & (type(result[4])==list),
                    f"do_simplex should return 2 tensors and 3 lists in this setting ({mod}, {d}). got {result}")

    
class TestWithDoSimplex(unittest.TestCase):
    """
    This class tests other funktions, using do_simplex, because it handles the more complex input variables like classifier, dataset, ..
    It tests original_model(), compact_original_model() and reimplemented_model() in their basic (not ablation) funtionality from simplex_versions.py.
    Also the jacobians from different models and create_decompositions() from evaluation.py.

    """    
    
    @classmethod
    def setUpClass(self):
        print(10*"-" + "training our simplex" + 10*"-")
        
        models = [m.Model_Type.ORIGINAL, m.Model_Type.ORIGINAL_COMPACT, m.Model_Type.REIMPLEMENTED, m.Model_Type.REIMPLEMENTED]     # without ablation models, with extra model reimplemented(decomp=100)
        self.decomposition_size = 5
        decomp = [5,5,5,100] # reimplemented100
        self.corpus_size = 100
        self.test_size = 10
        self.test_id = 0
        self.random_seed = 42
        self.results = []
        self.cv = 0
        for model, d in zip(models, decomp):
            w, lr2, or2, j, d = m.do_simplex(
                model_type=model, 
                decomposition_size=d, 
                corpus_size = self.corpus_size,
                test_size = self.test_size,
                test_id = self.test_id,
                r_2_scores=True,
                random_dataloader=False, # we want the same sample set for each model to train on
                )
            self.results.append({"w": w, "lr2": lr2, "or2": or2, "jac": j, "dec":d})
            # weights, latent_r2_score, output_r2_score, jacobian, decompostions

        self.sample_id = 0
        self.most_imp = 0

        self.orig_decomp = self.results[0]["dec"]
        self.compact_decomp = self.results[1]["dec"]
        self.rempl_decomp = self.results[2]["dec"]
        self.rempl_decomp_100 = self.results[3]["dec"]
        
        self.orig_weights = [self.orig_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        self.compact_weights = [self.compact_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        self.reimpl_weights = [self.rempl_decomp[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]
        self.reimpl100_weights = [self.rempl_decomp_100[self.sample_id]["decomposition"][i]["c_weight"] for i in range(self.decomposition_size)]

        self.orig_c_ids = [self.orig_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]
        self.compact_c_ids = [self.compact_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]
        self.reimpl_c_ids = [self.rempl_decomp[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]
        self.reimpl100_c_ids = [self.rempl_decomp_100[self.sample_id]["decomposition"][i]["c_id"] for i in range(self.decomposition_size)]

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
        self.assertEqual(self.results[1]["jac"].shape[0], self.corpus_size, "jacobian of compact model have incorrect shape!")
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
        self.assertTrue(is_close_w_index(self.orig_weights, self.compact_weights, tolerance=0.0), "original_weights and compact_weights schould be identical, but arent")
        self.assertTrue(is_close_w_index(self.orig_weights, self.reimpl100_weights, tolerance=0.1), "original_weights and compact_weights schould be close, but arent")

        # decomposition needs to add up to ~100% 
        self.assertAlmostEqual(sum(self.orig_weights), 1.0, delta=0.01, msg="original decomposition weights do not add up to at least 99%")
        # no need to test compact_weights, they are identical to orig_weights
        self.assertAlmostEqual(sum(self.reimpl100_weights), 1.0, delta=0.25, msg="reimplemented decomposition weights do not add up to  at least 99%")
        
    def test_jacobians(self):
        # row max in same place?
        orig_comp, o_max, _ = jacobian_compare_score(self.results[0]["jac"][0][0], self.results[1]["jac"][0][0])
        comp_reimpl, c_max, _ = jacobian_compare_score(self.results[1]["jac"][0][0], self.results[2]["jac"][0][0])
        reimpl_orig, r_max, _ = jacobian_compare_score(self.results[2]["jac"][0][0], self.results[0]["jac"][0][0])
        self.assertLessEqual(orig_comp, 0.0, f"Jacobians differ significantly btw original and compact model. row max locations: max-index-score={orig_comp}. max_index orig={o_max}, compact={c_max}")
        self.assertLessEqual(comp_reimpl, 1.0, f"Jacobians differ significantly btw reimplemeted and compact model. row max locations: max-index-score={comp_reimpl}. max_index orig={r_max}, compact={c_max}")
        self.assertLessEqual(reimpl_orig, 1.0, f"Jacobians differ significantly btw original and reimplemented model. row max locations: max-index-score={reimpl_orig}. max_index orig={o_max}, compact={r_max}")

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
        
        print(3*">" + "QUALITY: comparing most important corpus img in decomp between models")
        # check if explainer id's for first img is same for all models
        self.assertEqual(self.orig_decomp[self.sample_id]["decomposition"][self.most_imp]["c_id"], self.compact_decomp[self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & compact simplex!")

        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[2]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & reimplemented simplex!")

        # check if same corpus-ids in decomposition
        self.assertListEqual(self.orig_c_ids, self.compact_c_ids, f"corpus id's in decomposition differ btw original and compact model: {self.orig_c_ids}, {self.compact_c_ids}")
        print(3*">" + "QUALITY: comparing corpus id's in decomp between original and reimplemented simplex")

    # TODO: maybe test class-distr of classification against class-distr of decomposition

if __name__ == "__main__":
    unittest.main()
    #test = UnitTests()
    #test.setUpClass()
    #test.test_shuffle_data_loader()
    #test.test_make_corpus()
    #test.test_do_simplex()

    #test = TestWithDoSimplex()
    #test.setUpClass()
    #test.test_jacobians()
    


# execute all tests via console from root dir using
# python -m tests.unittests