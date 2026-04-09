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
