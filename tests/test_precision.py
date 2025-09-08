import unittest
import math
from ssee_metrics.precision import structured_semantic_entity_evaluation_precision as ssee_precision
from ssee_metrics.semantic_utils import SentenceTransformerSemanticUtils


class Test_SSEE_Precision(unittest.TestCase):

    def setUp(self):
        self.stsu = SentenceTransformerSemanticUtils()

    def test_precision_no_candidate_entities(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = []
        T = 0.5
        self.assertEqual(math.isnan(ssee_precision(E_cand, E_gold, T, self.stsu)), True)

    def test_precision_no_gold_entities(self):
        E_gold = []
        E_cand = ["apple", "banana", "cherry"]
        T = 0.5
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 0.0

    def test_precision_all_candidate_entities_match(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["apple", "banana", "cherry"]
        T = 0.5
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 1.0

    def test_precision_some_candidate_entities_match(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["apple", "banana", "grape"]
        T = 0.7
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 2/3

    def test_precision_no_candidate_entities_match(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["grape", "kiwi", "mango"]
        T = 0.7
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 0.0

    def test_precision_mixed_matches_with_different_thresholds(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["apple", "banana", "grape"]
        T = 1.0
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 2/3

    def test_precision_duplicate_candidate_entities(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["apple", "apple", "banana"]
        T = 0.7
        assert ssee_precision(E_cand, E_gold, T, self.stsu) == 2/3

    def test_threshold_check(self):
        E_gold = ["apple", "banana", "cherry"]
        E_cand = ["apple", "apple", "banana"]
        T = 100
        with self.assertRaises(ValueError):
            ssee_precision(E_cand, E_gold, T, self.stsu)
