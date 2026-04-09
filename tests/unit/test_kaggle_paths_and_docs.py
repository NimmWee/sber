from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.benchmark_latency as benchmark_latency_cli
import scripts.run_ablation as run_ablation_cli
from utils.script_helpers import resolve_transformers_provider_config


class KagglePathsAndDocsTest(unittest.TestCase):
    def test_readme_mentions_kaggle_working_and_input_paths(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("/kaggle/working", readme)
        self.assertIn("/kaggle/input", readme)

    def test_kaggle_validation_doc_exists_with_required_sections(self) -> None:
        doc = Path("docs/KAGGLE_VALIDATION.md")
        self.assertTrue(doc.exists())
        text = doc.read_text(encoding="utf-8")
        self.assertIn("smoke-check", text.lower())
        self.assertIn("full validation", text.lower())
        self.assertIn("benchmark_latency.py", text)
        self.assertIn("run_ablation.py", text)
        self.assertIn("/kaggle/working", text)
        self.assertIn("/kaggle/input", text)

    @patch("scripts.benchmark_latency.TransformersTokenStatProvider")
    @patch("scripts.benchmark_latency.resolve_transformers_provider_config")
    @patch("scripts.benchmark_latency.benchmark_frozen_submission_latency")
    def test_latency_cli_creates_kaggle_report_dir(
        self,
        benchmark_latency,
        resolve_config,
        provider_cls,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "/kaggle/temp/GigaChat3"})()
        provider_cls.return_value = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            benchmark_latency.return_value = {
                "json_report_path": str(report_dir / "latency_report.json"),
                "markdown_report_path": str(report_dir / "latency_report.md"),
                "sample_size": 4,
                "total_ms": {"avg": 10.0},
            }
            with patch(
                "sys.argv",
                [
                    "benchmark_latency.py",
                    "--dataset-path",
                    "/kaggle/input/demo/private.csv",
                    "--report-dir",
                    str(report_dir),
                ],
            ):
                benchmark_latency_cli.main()

            self.assertTrue(report_dir.exists())

    @patch("scripts.run_ablation.TransformersTokenStatProvider")
    @patch("scripts.run_ablation.resolve_transformers_provider_config")
    @patch("scripts.run_ablation.run_internal_ablation_report")
    def test_ablation_cli_creates_kaggle_report_dir(
        self,
        run_ablation_report,
        resolve_config,
        provider_cls,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "/kaggle/temp/GigaChat3"})()
        provider_cls.return_value = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            run_ablation_report.return_value = {
                "json_report_path": str(report_dir / "ablation_report.json"),
                "markdown_report_path": str(report_dir / "ablation_report.md"),
                "best_variant": "full_blend",
            }
            with patch(
                "sys.argv",
                [
                    "run_ablation.py",
                    "--dataset-path",
                    "/kaggle/working/sber/data/processed/textual_training_dataset.jsonl",
                    "--report-dir",
                    str(report_dir),
                ],
            ):
                run_ablation_cli.main()

            self.assertTrue(report_dir.exists())

    def test_missing_provider_config_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            with self.assertRaisesRegex(FileNotFoundError, "token_stat_provider"):
                resolve_transformers_provider_config(project_root=root)


if __name__ == "__main__":
    unittest.main()
