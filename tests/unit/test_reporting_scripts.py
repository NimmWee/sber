from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.benchmark_latency as latency_cli
import scripts.run_ablation as ablation_cli


class ReportingScriptsTest(unittest.TestCase):
    @patch("scripts.benchmark_latency.TransformersTokenStatProvider")
    @patch("scripts.benchmark_latency.resolve_transformers_provider_config")
    @patch("scripts.benchmark_latency.benchmark_frozen_submission_latency")
    def test_latency_script_writes_report_paths(
        self,
        benchmark_latency,
        resolve_config,
        provider_cls,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "stub-model"})()
        provider_cls.return_value = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            benchmark_latency.return_value = {
                "json_report_path": str(report_dir / "latency_report.json"),
                "markdown_report_path": str(report_dir / "latency_report.md"),
                "avg_latency_ms": 100.0,
            }
            with patch(
                "sys.argv",
                [
                    "benchmark_latency.py",
                    "--dataset-path",
                    str(Path(temp_dir) / "sample.csv"),
                    "--report-dir",
                    str(report_dir),
                ],
            ):
                latency_cli.main()

            kwargs = benchmark_latency.call_args.kwargs
            self.assertEqual(kwargs["report_dir"], report_dir)

    @patch("scripts.run_ablation.TransformersTokenStatProvider")
    @patch("scripts.run_ablation.resolve_transformers_provider_config")
    @patch("scripts.run_ablation.run_internal_ablation_report")
    def test_ablation_script_wires_report_dir(
        self,
        run_ablation_report,
        resolve_config,
        provider_cls,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "stub-model"})()
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
                    str(Path(temp_dir) / "dataset.jsonl"),
                    "--artifact-dir",
                    str(Path(temp_dir) / "model"),
                    "--report-dir",
                    str(report_dir),
                ],
            ):
                ablation_cli.main()

            kwargs = run_ablation_report.call_args.kwargs
            self.assertEqual(kwargs["report_dir"], report_dir)


if __name__ == "__main__":
    unittest.main()
