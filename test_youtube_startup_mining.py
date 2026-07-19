import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import youtube_startup_mining as ysm


class YouTubeStartupMiningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = Path(__file__).with_name("config.json")
        cls.config = ysm.load_config(cls.config_path)

    def test_parse_iso8601_duration(self):
        self.assertEqual(ysm.parse_iso8601_duration("PT15M33S"), 933)
        self.assertEqual(ysm.parse_iso8601_duration("PT1H2M3S"), 3723)
        self.assertEqual(ysm.parse_iso8601_duration("P1DT2H"), 93600)

    def test_topic_and_sentiment(self):
        topic, labels, score = ysm.classify_topic(
            "스타트업이 시드 투자 유치에 성공했다",
            self.config["keyword_groups"],
        )
        self.assertEqual(topic, "투자·자금")
        self.assertIn("투자·자금", labels)
        self.assertGreater(score, 0)

        label, value, positive, negative = ysm.classify_sentiment(
            "정말 유익하고 좋은 영상 감사합니다",
            self.config["positive_words"],
            self.config["negative_words"],
        )
        self.assertEqual(label, "긍정")
        self.assertGreater(value, 0)
        self.assertGreater(positive, negative)

    def test_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            analysis_dir = root / "analysis"
            ysm.create_demo_data(raw_dir, self.config)
            result = ysm.analyze(raw_dir, analysis_dir, self.config, demo=True)

            self.assertGreater(result["video_count"], 20)
            self.assertGreater(result["comment_count"], 50)
            self.assertTrue((analysis_dir / "startup_youtube_analysis.xlsx").exists())
            self.assertTrue((analysis_dir / "startup_youtube_all_results.md").exists())
            self.assertTrue((analysis_dir / "startup_youtube_report.html").exists())
            self.assertTrue((analysis_dir / "tables" / "topic_summary.csv").exists())

            markdown = (analysis_dir / "startup_youtube_all_results.md").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn(ysm.AUTHOR_CREDIT, markdown)
            self.assertIn(ysm.APP_VERSION, markdown)
            self.assertIn("전체 분석 결과표", markdown)
            self.assertIn("기술통계", markdown)

            manifest = json.loads(
                (analysis_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["demo_data"])

            videos = pd.read_csv(analysis_dir / "videos_enriched.csv")
            self.assertIn("views_per_day", videos.columns)
            self.assertIn("topic_primary", videos.columns)


if __name__ == "__main__":
    unittest.main()
