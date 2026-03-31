from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.metrics import compute_pr_auc


def test_compute_pr_auc_from_labels_and_probabilities() -> None:
    labels = [1, 1, 0, 0]
    probabilities = [0.95, 0.9, 0.2, 0.1]

    pr_auc = compute_pr_auc(labels, probabilities)

    assert pr_auc == 1.0
