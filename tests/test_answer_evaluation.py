import unittest

from video_rag.evaluation.answer import evaluate_answers, exact_match, token_f1


class AnswerEvaluationTests(unittest.TestCase):
    def test_exact_match_ignores_spacing_and_case(self):
        self.assertEqual(exact_match("CLIP 模型", "clip模型"), 1.0)

    def test_token_f1_partial_match(self):
        self.assertGreater(token_f1("飞机在机场降落", "飞机在机场"), 0.0)

    def test_evaluate_answers(self):
        result = evaluate_answers({"q1": "CLIP"}, {"q1": "clip"})
        self.assertEqual(result["exact_match"], 1.0)

