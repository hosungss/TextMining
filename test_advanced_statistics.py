import tempfile
import unittest
from pathlib import Path

import pandas as pd

import youtube_startup_mining as ysm

try:
    import advanced_statistics as adv
except ImportError:
    adv = None


@unittest.skipIf(adv is None, "고급 통계 라이브러리가 설치되지 않았습니다.")
class AdvancedStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ysm.load_config(Path(__file__).with_name("config.json"))
        cls.config["advanced_methods"] = list(adv.METHOD_CATEGORIES)
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            ysm.create_demo_data(raw_dir, cls.config)
            cls.videos = ysm.enrich_videos(
                pd.read_csv(raw_dir / "videos.csv", encoding="utf-8-sig"), cls.config
            )
            cls.comments = ysm.enrich_comments(
                pd.read_csv(raw_dir / "comments.csv", encoding="utf-8-sig"), cls.config
            )

    def test_advanced_suite_returns_major_method_families(self):
        results = adv.run_advanced_statistics(
            self.videos, self.comments, self.config
        )
        expected = {
            "normality_tests",
            "two_group_tests",
            "multi_group_tests",
            "full_correlations",
            "contingency_tests",
            "regression_models",
            "time_series_tests",
            "pca_loadings",
            "cluster_profiles",
            "outlier_summary",
            "power_analysis",
            "method_catalog",
        }
        self.assertTrue(expected.issubset(results))
        self.assertFalse(results["normality_tests"].empty)
        self.assertFalse(results["full_correlations"].empty)
        self.assertFalse(results["method_catalog"].empty)

    def test_pvalues_have_multiple_testing_corrections(self):
        result = adv.full_correlations(self.videos)
        self.assertIn("p_holm", result.columns)
        self.assertIn("q_bh", result.columns)
        self.assertTrue(result["q_bh"].between(0, 1).all())

    def test_api_key_is_not_part_of_analysis_configuration(self):
        self.assertNotIn("api_key", self.config)
        self.assertNotIn("YOUTUBE_API_KEY", self.config)


if __name__ == "__main__":
    unittest.main()
