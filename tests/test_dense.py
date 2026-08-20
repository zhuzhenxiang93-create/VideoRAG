import unittest

import numpy as np

from video_rag.retrieval.faiss_dense import (
    ClipVisionRetriever,
    normalize_rows,
    unwrap_model_features,
)


class DenseVectorTests(unittest.TestCase):
    def test_normalize_rows_returns_float32_unit_vectors(self):
        vectors = normalize_rows(np.array([[3, 4], [0, 2]], dtype=np.float64))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)

    def test_normalize_rows_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            normalize_rows(np.zeros((1, 2), dtype=np.float32))
    def test_unwrap_model_features_supports_structured_outputs(self):
        class Output:
            pooler_output = np.array([[1.0, 2.0]], dtype=np.float32)

        result = unwrap_model_features(Output())
        np.testing.assert_array_equal(result, Output.pooler_output)

    def test_unwrap_model_features_keeps_tensor_like_output(self):
        output = np.array([[3.0, 4.0]], dtype=np.float32)
        self.assertIs(unwrap_model_features(output), output)

    def test_chinese_clip_uses_separate_index_name(self):
        retriever = ClipVisionRetriever("OFA-Sys/chinese-clip-vit-base-patch16")
        self.assertEqual(retriever.name, "vision_dense_zh")

    def test_openai_clip_keeps_original_index_name(self):
        retriever = ClipVisionRetriever("openai/clip-vit-large-patch14")
        self.assertEqual(retriever.name, "vision_dense")
