import inspect
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader
import unittest

sys.path.insert(0, "")

from original_code.src.simplexai.models.image_recognition import MnistClassifier
import src.classifier_versions as c
import src.evaluation as e
import src.main as m
import src.simplex_versions as s
from src.classifier.CatsAndDogsClassifier import CatsandDogsClassifier
from src.classifier.HeartfailureClassifier import HeartFailureClassifier
from src.datasets.cats_and_dogs_dataset import CandDDataSet
from src.utils.image_finder_cats_and_dogs import LABEL, get_images
from src.utils.corpus_creator import make_corpus
from src.utils.utlis import is_close_w_index, plot_jacobians_grayscale, print_jacobians_with_img

class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print(10*"-" + "Unit tests" + 10*"-")
        self.random_seed = 42
        self.decomposition_size = 5
        self.corpus_size = 10
        self.test_size = 1
        self.test_id = 0


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
        tests TODO check if all loaders return classifier and correct format and types for corpus and test sets.
        """
        print(3*">" + "testing data loader format and type")
        loaders = [c.train_or_load_mnist, c.train_or_load_heartfailure_model , c.train_or_load_CaD_model]
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

        for loader in [c.train_or_load_mnist]:#TODO, c.train_or_load_CaD_model, c.train_or_load_heartfailure_model]:

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
    
    # test make_corpus
    # test corps loader class distribution
    def test_make_corpus(self):
        test_files = [f"{i}_test" for i in range(1000)]   
        test_labels = [i%2 for i in range(100)]
        dataset = CandDDataSet(image_paths=test_files, labels=test_labels, transform=None)
        datalaoder = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        corpus = make_corpus(datalaoder)
        print("TODO: create with lucas")
    
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



    # edge cases für input var (testid > testset) size of corpus&test (10, 100, 1000) 

    # exceptions datset & model
            
    # exception (decompsition > corpus)
            
    def _test_do_simplex():
        print("this should not be executed")
        # if 4x false, result should be tuple[torch.Tensor, None, None, None, None]
        result = m.do_simplex(
                    model_type=m.Model_Type.ORIGINAL,
                    dataset=m.Dataset.MNIST,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False,
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )
        
        # return weights, latent_r2_score, output_r2_score, jacobian, decompostions
        # tuple[torch.Tensor, list[float], list[float], torch.Tensor, list[dict]]
        # for mnist, all model types should give valid return val
        for d in m.Dataset.MNIST:
            for mod in m.Model_Type:
                result = m.do_simplex(
                    model_type=mod,
                    dataset=d,
                    cv=0,
                    decomposition_size=3,
                    corpus_size=10,
                    test_size=1,
                    test_id=0,
                    print_jacobians=False,
                    r_2_scores=False,
                    decompose=False,
                    random_dataloader=False
                )

    
class TestWithDoSimplex(unittest.TestCase):
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
        #TODO: for decomposition = 100
        #self.assertTrue(is_close_w_index(self.orig_weights, self.reimpl_weights))

        # decomposition needs to add up to ~100% 
        self.assertAlmostEqual(sum(self.orig_weights), 1.0, delta=0.01, msg="original decomposition weights do not add up to 99%")
        self.assertAlmostEqual(sum(self.reimpl_weights), 1.0, delta=0.25, msg="reimplemented decomposition weights do not add up to 99%") #TODO: dec_size=100

        #TODO: explain why decomposition 100!
        
    def test_jacobians(self):
        print("TODO: implement test_jacobians")# TODO: implement
        # test original jacobian method against ours
        plot_jacobians_grayscale(self.results[0]["jac"][0][0])
        plot_jacobians_grayscale(self.results[2]["jac"][0][0])
        plot_jacobians_grayscale(self.results[2]["jac"][0][0]-self.results[0]["jac"][0][0])
        #print_jacobians_with_img()#TODO test this
        
        # element wise mean_square_error
        # row max in same place?
        self.assertTrue(torch.equal(self.results[0]["jac"][0], self.results[2]["jac"][0]), "first reimplemented jacobian differs from first original jacobian")

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

        print(3*">" + "checking most important corpus id in decomposition") #TODO: duplacate to dataloader?
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
        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[1]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & compact simplex!") #TODO: change from int to model

        self.assertEqual(self.results[0]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], self.results[2]["dec"][self.sample_id]["decomposition"][self.most_imp]["c_id"], msg="most important explainer differs btw original & reimplemented simplex!")

        # check if same corpus-ids in decomposition
        self.assertListEqual(self.orig_c_ids, self.compact_c_ids, f"corpus id's in decomposition differ btw original and compact model: {self.orig_c_ids}, {self.compact_c_ids}")
        print(3*">" + "QUALITY: comparing corpus id's in decomp between original and reimplemented simplex")
        self.assertListEqual(self.orig_c_ids, self.reimpl_c_ids, f"QUALITY: corpus id's in decomposition differ btw original and reimplemented model: {self.orig_c_ids}, {self.reimpl_c_ids}") #TODO: decomposition 100
        #TODO unocmment test and explaiin why


   
    # maybe test class-distr of classification against class-distr of decomposition
        
    # does test_size influence simplex performance?

if __name__ == "__main__":
    unittest.main()
    #test = UnitTests()
    #test.setUpClass()
    #test.test_data_loaders()


# execute all tests via console from root dir using
# python -m tests.unittests