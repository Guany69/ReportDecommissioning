"""Offline training and evaluation for the duplicate classifier.

Everything here runs locally against local files — nothing is uploaded, and no
hosted model service is contacted. This package is *not* imported by the runtime
engine or the FastAPI service; production only needs `report_cleanup.ml`.

Workflow (details in the README):

    python -m training.generate_pairs           -> reviewer CSV with a blank label column
    <a human labels it 1 / 0>
    python -m training.train_duplicate_model    -> models/duplicate_model.pt
    python -m training.evaluate_duplicate_model -> baseline vs. PyTorch metrics

The labels must come from human review. Deriving them from the existing weighted
similarity formula and then scoring the model against those same derived labels
would only measure how well the network imitates the heuristic, which is not
evidence that either one is correct.
"""

# These names are part of the reviewer-CSV contract. Report UIDs are strings in
# the training package even though the current pipeline happens to allocate
# integer UIDs: treating them as opaque identifiers avoids losing leading zeroes
# if a future Workday export supplies a stable report reference ID.
REPORT_A_ID_COLUMN = "report_uid_a"
REPORT_B_ID_COLUMN = "report_uid_b"
REPORT_A_NAME_COLUMN = "report_name_a"
REPORT_B_NAME_COLUMN = "report_name_b"
LABEL_COLUMN = "label"
BASELINE_SCORE_COLUMN = "baseline_similarity"
BASELINE_PREDICTION_COLUMN = "baseline_prediction"
BASELINE_RELATIONSHIP_COLUMN = "baseline_relationship"
BASELINE_THRESHOLD_COLUMN = "baseline_decision_threshold"

CANONICAL_PAIR_COLUMNS = (
    REPORT_A_ID_COLUMN,
    REPORT_A_NAME_COLUMN,
    REPORT_B_ID_COLUMN,
    REPORT_B_NAME_COLUMN,
)

__all__ = [
    "BASELINE_PREDICTION_COLUMN",
    "BASELINE_RELATIONSHIP_COLUMN",
    "BASELINE_SCORE_COLUMN",
    "BASELINE_THRESHOLD_COLUMN",
    "CANONICAL_PAIR_COLUMNS",
    "LABEL_COLUMN",
    "REPORT_A_ID_COLUMN",
    "REPORT_A_NAME_COLUMN",
    "REPORT_B_ID_COLUMN",
    "REPORT_B_NAME_COLUMN",
]
