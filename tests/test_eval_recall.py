"""The recall report of eval/recall.py and the question file: no database, no model."""

import json
from pathlib import Path

from eval.recall import QUESTIONS, report

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_recall_is_reported_per_set_and_language():
    questions = [
        {"set": "base", "expected_doc_id": "a"},
        {"set": "base", "expected_doc_id": "b"},
        {"set": "exact_terms", "expected_doc_id": "c"},
    ]
    text = report(questions, {"en": [True, True, False], "uk (as is)": [True, False, False]})

    assert "| Queries | base (2) | exact_terms (1) |" in text
    assert "| en | 2/2 (100%) | 0/1 (0%) |" in text
    assert "missed, uk (as is): b [base], c [exact_terms]" in text


def test_every_question_has_a_set_both_languages_and_an_existing_document():
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    assert {q["set"] for q in questions} == {"base", "exact_terms"}
    for q in questions:
        assert q["question_en"] and q["question_uk"]
        assert (CORPUS / f"{q['expected_doc_id']}.md").exists(), q["expected_doc_id"]
