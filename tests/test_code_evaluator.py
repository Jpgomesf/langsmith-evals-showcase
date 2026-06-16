"""Deterministic unit tests for the multi-key field-extraction evaluator."""

from __future__ import annotations

from evals_showcase.evaluators.code import (
    FieldSpec,
    compare_field,
    make_field_extraction_evaluator,
    score_fields,
)

FIELDS = [
    FieldSpec("merchant", "text"),
    FieldSpec("amount", "numeric", tolerance=0.01),
    FieldSpec("date", "exact"),
    FieldSpec("category", "exact"),
]


def test_compare_text_is_case_insensitive():
    assert compare_field(FieldSpec("m", "text"), "Whole Foods", "whole foods") == 1
    assert compare_field(FieldSpec("m", "text"), "Costco", "Walmart") == 0


def test_compare_numeric_tolerance():
    spec = FieldSpec("amount", "numeric", tolerance=0.01)
    assert compare_field(spec, 45.99, 45.99) == 1
    assert compare_field(spec, 45.991, 45.99) == 1  # within tolerance
    assert compare_field(spec, 46.50, 45.99) == 0
    assert compare_field(spec, None, 45.99) == 0  # missing value, no crash


def test_compare_exact():
    assert compare_field(FieldSpec("date", "exact"), "2026-03-14", "2026-03-14") == 1
    assert compare_field(FieldSpec("date", "exact"), "2026-03-15", "2026-03-14") == 0


def test_score_fields_all_correct():
    out = {
        "merchant": "Whole Foods",
        "amount": 45.99,
        "date": "2026-03-14",
        "category": "groceries",
    }
    results = score_fields(FIELDS, out, out)["results"]
    by_key = {r["key"]: r["score"] for r in results}
    assert by_key["all_fields_present"] == 1
    assert by_key["field_accuracy"] == 1.0
    assert by_key["merchant_match"] == by_key["amount_match"] == 1


def test_score_fields_partial_and_accuracy_is_mean():
    ref = {
        "merchant": "Whole Foods",
        "amount": 45.99,
        "date": "2026-03-14",
        "category": "groceries",
    }
    out = {"merchant": "whole foods", "amount": 50.00, "date": "2026-03-14", "category": "dining"}
    by_key = {r["key"]: r["score"] for r in score_fields(FIELDS, out, ref)["results"]}
    # merchant ✓ (text), amount ✗, date ✓, category ✗ -> 2/4
    assert by_key["field_accuracy"] == 0.5
    assert by_key["amount_match"] == 0
    assert by_key["merchant_match"] == 1


def test_score_fields_missing_field_flags_not_present():
    ref = {
        "merchant": "Whole Foods",
        "amount": 45.99,
        "date": "2026-03-14",
        "category": "groceries",
    }
    out = {"merchant": "", "amount": 45.99, "date": "2026-03-14", "category": "groceries"}
    by_key = {r["key"]: r["score"] for r in score_fields(FIELDS, out, ref)["results"]}
    assert by_key["all_fields_present"] == 0
    assert by_key["merchant_match"] == 0


def test_evaluator_returns_results_envelope():
    out = {"merchant": "Nike", "amount": 120.0, "date": "2026-03-22", "category": "shopping"}
    payload = make_field_extraction_evaluator(FIELDS)(out, out)
    keys = {r["key"] for r in payload["results"]}
    assert keys == {
        "all_fields_present",
        "merchant_match",
        "amount_match",
        "date_match",
        "category_match",
        "field_accuracy",
    }
