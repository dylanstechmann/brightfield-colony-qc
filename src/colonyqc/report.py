"""JSON QC report. The disclaimer is part of the schema, not a comment."""

from __future__ import annotations

from colonyqc.features import FEATURE_NAMES

DISCLAIMER = (
    "Research morphology triage from a brightfield field. "
    "Not a karyotype, STR identity assay, residual-reprogramming-factor assay, "
    "mycoplasma test, potency assay, or lot-release criterion. "
    "A contamination flag means 'look at the plate and run a validated assay', "
    "not a diagnosis. Not for administration of any cell product to humans."
)


def build_report(label: str, proba: dict[str, float], features: list[float]) -> dict:
    flags = []
    if proba.get("contamination_suspect", 0.0) >= 0.35:
        flags.append("contamination_triage")
    if proba.get("debris", 0.0) >= 0.5:
        flags.append("mostly_debris")
    if label == "differentiating":
        flags.append("morphology_not_undifferentiated")
    return {
        "disclaimer": DISCLAIMER,
        "call": label,
        "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
        "features": {name: round(float(val), 5) for name, val in zip(FEATURE_NAMES, features)},
        "flags": flags,
        "next_human_step": _next(flags),
    }


def _next(flags: list[str]) -> str:
    if "contamination_triage" in flags:
        return "Quarantine the culture. Inspect under phase contrast. Run a validated mycoplasma assay before any further use."
    if "mostly_debris" in flags:
        return "Field looks empty or full of debris. Check focus, seeding, and whether the vessel was fed."
    if "morphology_not_undifferentiated" in flags:
        return "Morphology is not a compact undifferentiated colony. Confirm the intended fate with a marker assay before expanding."
    return "Morphology is consistent with a compact colony on this model. Still confirm identity and sterility on the lab's schedule."
