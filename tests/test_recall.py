import unittest
import math
from ssee_metrics.recall import structured_semantic_entity_evaluation_recall as ssee_recall
from ssee_metrics.semantic_utils import SentenceTransformerSemanticUtils

class Test_SSEE_Recall(unittest.TestCase):

    def setUp(self):
        self.stsu = SentenceTransformerSemanticUtils()

    def test_perfect_match(self):
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = ["Apple", "Orange", "Lemon"]
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 1.0)

    def test_partial_match(self):
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = ["Apple", "Orange"]
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 2/3)

    def test_no_match(self):
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = ["Plum", "Grape", "Melon"]
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 0.0)

    def test_threshold_match(self):
        self.sem_sim = lambda x, y: 0.5 if x == y else 0  # Adjust similarity function
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = ["Apple", "Orange", "Lemon"]
        T = 0.5
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 1.0)

    def test_empty_candidates(self):
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = []
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 0.0)

    def test_empty_gold(self):
        E_gold = []
        E_cand = ["Apple", "Orange", "Lemon"]
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(math.isnan(recall), True)

    def test_mixed_match(self):
        E_gold = ["Apple", "Orange", "Lemon"]
        E_cand = ["Apple", "Plum", "Lemon"]
        T = 1.0
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 2/3)

    def test_semantic_match_above_threshold(self):
        E_gold = ["Apple", "Orange"]
        E_cand = ["Cooking Apple", "Sweet Orange"]
        T = 0.6
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 1.0)

    def test_only_match_once(self):
        E_gold = ["Apple", "Orange", "Cooking Apple"]
        E_cand = ["Cooking Apple", "Sweet Orange", "Cows"]
        T = 0.7
        recall = ssee_recall(E_cand, E_gold, T, self.stsu)
        self.assertEqual(recall, 2/3)
