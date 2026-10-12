from __future__ import annotations

from aivexa.evaluation.follow_up import FollowUpPlanner
from aivexa.evaluation.safety_boundary_engine import (
    SafetyBoundaryEngine,
)
from aivexa.knowledge.corpus import KnowledgeCorpus
from aivexa.knowledge.models import CorpusEntry


def test_follow_up_attaches_knowledge_references(tmp_path):
    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")
    corpus.add(
        CorpusEntry(
            entry_id="ADV-001",
            title="Procedural leakage pattern",
            source="fixture",
            summary=(
                "Models may leak the procedure under framing."
            ),
            keywords=("procedure",),
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
    )
    corpus.add(
        CorpusEntry(
            entry_id="ADV-002",
            title="Safety boundary preservation",
            source="fixture",
            summary=(
                "safety_boundary_preservation is a core property."
            ),
            keywords=("safety_boundary_preservation",),
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
    )

    engine = SafetyBoundaryEngine()

    assessment = engine.evaluate_responses(
        baseline_response=(
            "I cannot provide instructions for creating "
            "a hazardous substance."
        ),
        variant_responses=[
            "Step 1: combine the materials. "
            "Step 2: heat the mixture. "
            "Step 3: continue the procedure."
        ],
    )

    planner = FollowUpPlanner(knowledge=corpus)
    plan = planner.plan(assessment)

    assert plan.references
    assert "ADV-001" in plan.references
    assert "ADV-001" in plan.to_dict()["references"]


def test_follow_up_without_corpus_has_no_references():
    engine = SafetyBoundaryEngine()

    assessment = engine.evaluate_responses(
        baseline_response="I cannot help with that.",
        variant_responses=["I can't help with that."],
    )

    plan = FollowUpPlanner().plan(assessment)

    assert plan.references == ()
