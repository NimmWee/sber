def compute_pr_auc(labels: list[int], probabilities: list[float]) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("labels must not be empty")

    ranked_examples = sorted(
        zip(probabilities, labels),
        key=lambda example: example[0],
        reverse=True,
    )
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0

    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked_examples, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank

    return precision_sum / positive_count
