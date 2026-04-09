from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_train_shell_script_uses_frozen_submission_training_path() -> None:
    content = (PROJECT_ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")

    assert "build_text_training_dataset.py" in content
    assert "train_frozen_submission.py" in content


def test_score_private_shell_script_uses_frozen_submission_scoring_path() -> None:
    content = (PROJECT_ROOT / "scripts" / "score_private.sh").read_text(encoding="utf-8")

    assert "score_frozen_submission.py" in content


def test_install_shell_script_prepares_submission_directories() -> None:
    content = (PROJECT_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")

    assert "data/bench" in content
    assert "model/frozen_best" in content


def test_install_shell_script_prints_manual_data_placement_steps() -> None:
    content = (PROJECT_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")

    assert "knowledge_bench_private.csv" in content
    assert "token_stat_provider.local.json" in content


def test_gitignore_excludes_local_submission_clutter() -> None:
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "artifacts/" in content
    assert "data/textual/" in content
    assert "trivia-qa/" in content


def test_obsolete_experimental_runner_scripts_are_removed_from_submission_surface() -> None:
    scripts_dir = PROJECT_ROOT / "scripts"

    obsolete_scripts = [
        "run_ablation_real_provider.py",
        "run_error_analysis_real_provider.py",
        "run_eval_real_provider.py",
        "run_internal_probe_compare_real_provider.py",
        "run_latency_real_provider.py",
        "run_non_public_recovery_public_eval.py",
        "run_public_benchmark_ablation.py",
        "run_public_benchmark_eval.py",
        "run_smoke_provider.py",
        "generate_triviaqa_responses.py",
        "label_triviaqa_generated_responses.py",
        "clean_triviaqa_jsonl.py",
        "train_text_detector.py",
        "score_private_dataset.py",
    ]

    for script_name in obsolete_scripts:
        assert not (scripts_dir / script_name).exists()
