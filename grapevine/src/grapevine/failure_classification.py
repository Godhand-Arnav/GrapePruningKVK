"""PRD §5.3 — Failure Classification.

Buckets a single test-image result into F1-F5 (or none). Used by the
eval/failure-analysis step (not part of the per-image inference pipeline
itself) to produce the frequency table required by PRD §11 item 8.

| Code | Failure                              |
|------|----------------------------------------|
| F1   | Missing thin shoot (<3px)              |
| F2   | Broken mask (gap in structure)          |
| F3   | Wrong class (cane labeled shoot)        |
| F4   | Merged vines (two vines in one graph)   |
| F5   | False positive (background -> vine)     |
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class FailureReport:
    image: str
    codes: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    def add(self, code: str, detail: str) -> None:
        self.codes.append(code)
        self.detail.append(detail)


def classify_image(
    image_name: str,
    predicted_graph: nx.Graph,
    hierarchy_violations: list[dict],
    gt_shoot_count: int | None = None,
    pred_shoot_count: int | None = None,
    n_components_with_trunk: int | None = None,
    false_positive_count: int = 0,
) -> FailureReport:
    report = FailureReport(image=image_name)

    # F1 — missing thin shoots: predicted shoot count well below ground truth
    if gt_shoot_count is not None and pred_shoot_count is not None and gt_shoot_count > 0:
        missing_frac = max(0.0, (gt_shoot_count - pred_shoot_count) / gt_shoot_count)
        if missing_frac > 0:
            report.add("F1", f"{missing_frac:.0%} of ground-truth shoots missing "
                              f"({pred_shoot_count}/{gt_shoot_count} predicted).")

    # F2 — broken mask: isolated 1-node "islands" (skeleton fragment with no
    # edges) or components with no root-reachable path to a leaf
    isolated_nodes = [n for n in predicted_graph.nodes if predicted_graph.degree(n) == 0]
    if isolated_nodes:
        report.add("F2", f"{len(isolated_nodes)} disconnected skeleton fragment(s) found.")

    # F3 — wrong class: hierarchy violations flagged during tracing that
    # indicate a probable mislabel (cane before cordon, etc.)
    for v in hierarchy_violations:
        if v["code"] == "F3":
            report.add("F3", v["detail"])

    # F4 — merged vines: >1 trunk-bearing component, or a hierarchy violation
    # explicitly flagged as F4 during trace
    if n_components_with_trunk is not None and n_components_with_trunk > 1:
        report.add("F4", f"{n_components_with_trunk} separate trunk-bearing components "
                          f"detected in one image — check for merged vines.")
    for v in hierarchy_violations:
        if v["code"] == "F4":
            report.add("F4", v["detail"])

    # F5 — false positives: background misclassified as vine structure
    if false_positive_count > 0:
        report.add("F5", f"{false_positive_count} false-positive detections "
                          f"(background classified as vine structure).")

    return report


def summarize(reports: list[FailureReport]) -> dict:
    """Aggregate frequency table across the test set, per PRD §5.3 / §11.8."""
    n = len(reports)
    counts = {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "F5": 0}
    for r in reports:
        for code in set(r.codes):  # count each failure type once per image
            counts[code] += 1
    return {
        "n_test_images": n,
        "counts": counts,
        "rates": {k: round(v / n, 3) if n else 0.0 for k, v in counts.items()},
    }
