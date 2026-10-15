from __future__ import annotations

from deepheal.paths import ARTICLE_ROOT


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSelection:
    model: str
    source: str
    selection_mode: str


def _read_summary(summary_json: Path | str | None) -> dict[str, Any]:
    if summary_json is None:
        return {}
    path = Path(summary_json)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _predictor_best(predictor: object) -> str:
    return _clean(getattr(predictor, "model_best", ""))


def _predictor_model_names(predictor: object) -> list[str]:
    model_names = getattr(predictor, "model_names", None)
    if callable(model_names):
        return [str(name) for name in model_names()]
    return []


def _select_first(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for model, source in candidates:
        cleaned = _clean(model)
        if cleaned:
            return cleaned, source
    return "", ""


def resolve_model_selection(
    predictor: object,
    explicit_model: str = "",
    summary_json: Path | str | None = None,
    selection_mode: str = "summary_best",
) -> ModelSelection:
    summary = _read_summary(summary_json)
    mode = _clean(selection_mode) or "summary_best"
    explicit = _clean(explicit_model)

    if mode == "summary_best":
        candidates = [
            (explicit, "explicit"),
            (_clean(summary.get("best_model")), "summary.best_model"),
            (_predictor_best(predictor), "predictor.model_best"),
        ]
    elif mode == "external_best":
        candidates = [
            (explicit, "explicit"),
            (_clean(summary.get("best_single_model_by_existing_code_rule")), "summary.best_single_model_by_existing_code_rule"),
            (_clean(summary.get("best_model")), "summary.best_model"),
            (_predictor_best(predictor), "predictor.model_best"),
        ]
    elif mode == "predictor_best":
        candidates = [
            (explicit, "explicit"),
            (_predictor_best(predictor), "predictor.model_best"),
            (_clean(summary.get("best_model")), "summary.best_model"),
        ]
    else:
        raise ValueError(f"Unsupported AutoGluon model selection mode: {selection_mode}")

    model, source = _select_first(candidates)
    if not model:
        raise ValueError("Could not resolve an AutoGluon model from explicit input, summary JSON, or predictor.model_best.")

    available_models = _predictor_model_names(predictor)
    if available_models and model not in available_models:
        raise ValueError(
            f"Resolved AutoGluon model '{model}' from {source}, but predictor only contains: {', '.join(available_models)}"
        )
    return ModelSelection(model=model, source=source, selection_mode=mode)


def model_label_from_predictions(predictions: object, fallback: str) -> str:
    get_value = getattr(predictions, "get", None)
    raw_values = get_value("autogluon_model", None) if callable(get_value) else None
    if raw_values is None:
        return fallback

    unique: list[str] = []
    for value in raw_values:
        model = _clean(value)
        if model and model not in unique:
            unique.append(model)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "+".join(unique)
    return fallback
