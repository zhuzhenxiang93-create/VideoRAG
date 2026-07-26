import unittest

import numpy as np

from video_rag.retrieval.faiss_dense import normalize_rows


class DenseVectorTests(unittest.TestCase):
    def test_normalize_rows_returns_float32_unit_vectors(self):
        vectors = normalize_rows(np.array([[3, 4], [0, 2]], dtype=np.float64))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)

    def test_normalize_rows_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            normalize_rows(np.zeros((1, 2), dtype=np.float32))

