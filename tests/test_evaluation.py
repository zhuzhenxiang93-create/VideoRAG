import unittest

from video_rag.evaluation.retrieval import evaluate_retrieval


class EvaluationTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = evaluate_retrieval(
            predictions={"q1": ["x", "a"], "q2": ["b"]},
            ground_truth={"q1": ["a"], "q2": ["b"]},
            cutoffs=(1, 2),
        )
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
