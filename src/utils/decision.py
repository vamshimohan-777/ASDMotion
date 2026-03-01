# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

from dataclasses import dataclass


@dataclass
class DecisionResult:
    decision: str
    prob_raw: float
    prob_calibrated: float
    quality_score: float
    threshold_used: float
    reasons: list
    abstained: bool


def _positive_stage(prob_cal: float, high_thr: float) -> str:
    # Split [high_thr, 1.0] into 3 equal bands for LOW/MEDIUM/HIGH positive staging.
    span = max(1e-6, 1.0 - float(high_thr))
    low_to_med = float(high_thr) + span / 3.0
    med_to_high = float(high_thr) + 2.0 * span / 3.0

    if prob_cal >= med_to_high:
        return "HIGH"
    if prob_cal >= low_to_med:
        return "MEDIUM"
    return "LOW"


def _chance_label(stage: str) -> str:
    if stage == "HIGH":
        return "HIGH CHANCES OF ASD"
    if stage == "MEDIUM":
        return "MEDIUM CHANCES OF ASD"
    return "LOW CHANCES OF ASD"


def make_decision(prob_raw: float, prob_cal: float, quality_score: float,
                  quality_threshold: float, low_thr: float, high_thr: float) -> DecisionResult:
    reasons = []
    if quality_score < quality_threshold:
        reasons.append("Quality below threshold")
        return DecisionResult(
            decision="LOW QUALITY VIDEO/FALSE VIDEO UPLOAD",
            prob_raw=prob_raw,
            prob_calibrated=prob_cal,
            quality_score=quality_score,
            threshold_used=quality_threshold,
            reasons=reasons,
            abstained=True,
        )

    reasons.append("Quality ok")
    if prob_cal >= high_thr:
        stage = _positive_stage(prob_cal, high_thr)
        reasons.append("Calibrated prob above positive threshold")
        reasons.append(f"ASD chance stage: {stage}")
        return DecisionResult(
            decision=_chance_label(stage),
            prob_raw=prob_raw,
            prob_calibrated=prob_cal,
            quality_score=quality_score,
            threshold_used=high_thr,
            reasons=reasons,
            abstained=False,
        )
    if prob_cal <= low_thr:
        reasons.append("Calibrated prob below negative threshold")
        return DecisionResult(
            decision="NEGATIVE",
            prob_raw=prob_raw,
            prob_calibrated=prob_cal,
            quality_score=quality_score,
            threshold_used=low_thr,
            reasons=reasons,
            abstained=False,
        )

    reasons.append("Calibrated prob in recheck range")
    return DecisionResult(
        decision="NEEDS RECHECKING",
        prob_raw=prob_raw,
        prob_calibrated=prob_cal,
        quality_score=quality_score,
        threshold_used=0.5,
        reasons=reasons,
        abstained=False,
    )

