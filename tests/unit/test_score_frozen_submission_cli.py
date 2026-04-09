from __future__ import annotations

import unittest
from unittest.mock import patch

import scripts.score_frozen_submission as cli


class ScoreFrozenSubmissionCliTest(unittest.TestCase):
    @patch("scripts.score_frozen_submission.score_private_frozen_submission")
    @patch("scripts.score_frozen_submission.TransformersTokenStatProvider")
    @patch("scripts.score_frozen_submission.resolve_frozen_submission_config")
    @patch("scripts.score_frozen_submission.resolve_transformers_provider_config")
    def test_cli_defaults_to_probability_mode(
        self,
        resolve_config,
        resolve_submission_config,
        provider_cls,
        score_submission,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "stub-model"})()
        resolve_submission_config.return_value = {"serving_threshold": 0.3}
        provider_cls.return_value = object()
        score_submission.return_value = {
            "sample_size": 1,
            "output_path": "out.csv",
            "metadata": {"blend_version": "frozen_best_v1"},
        }

        with patch("sys.argv", ["score_frozen_submission.py"]):
            cli.main()

        kwargs = score_submission.call_args.kwargs
        self.assertEqual(kwargs["output_mode"], "probability")
        self.assertEqual(kwargs["label_threshold"], 0.3)

    @patch("scripts.score_frozen_submission.score_private_frozen_submission")
    @patch("scripts.score_frozen_submission.TransformersTokenStatProvider")
    @patch("scripts.score_frozen_submission.resolve_frozen_submission_config")
    @patch("scripts.score_frozen_submission.resolve_transformers_provider_config")
    def test_cli_uses_submission_threshold_when_not_overridden(
        self,
        resolve_config,
        resolve_submission_config,
        provider_cls,
        score_submission,
    ) -> None:
        resolve_config.return_value = type("Config", (), {"model_source": "stub-model"})()
        resolve_submission_config.return_value = {"serving_threshold": 0.42}
        provider_cls.return_value = object()
        score_submission.return_value = {
            "sample_size": 1,
            "output_path": "out.csv",
            "metadata": {"blend_version": "frozen_best_v1"},
        }

        with patch("sys.argv", ["score_frozen_submission.py", "--output-mode", "boolean"]):
            cli.main()

        kwargs = score_submission.call_args.kwargs
        self.assertEqual(kwargs["output_mode"], "boolean")
        self.assertEqual(kwargs["label_threshold"], 0.42)


if __name__ == "__main__":
    unittest.main()
