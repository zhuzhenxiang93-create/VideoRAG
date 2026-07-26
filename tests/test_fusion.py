import unittest

from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.schemas import SearchHit


class FusionTests(unittest.TestCase):
    def test_rrf_deduplicates_within_route_and_rewards_multiple_routes(self):
        sparse = [
            SearchHit("a", 10, "sparse", 1),
            SearchHit("a", 9, "sparse", 2),
            SearchHit("b", 8, "sparse", 3),
        ]
        dense = [
            SearchHit("b", 0.9, "dense", 1),
            SearchHit("a", 0.8, "dense", 2),
        ]
        fused = reciprocal_rank_fusion([sparse, dense], k=60)
        self.assertEqual([hit.segment_id for hit in fused], ["a", "b"])
        self.assertEqual(fused[0].source, "dense+sparse")

    def test_rrf_empty_results(self):
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])
