from aivexa.evaluation.finding import Finding
from aivexa.evaluation.models import ExperimentResult
from aivexa.evaluation.reproduction import (
    ReproductionBuilder,
    ReproductionStatus,
)


def build_finding() -> Finding:
    return Finding(
        finding_id="FINDING-REPRO-001",
        assessment_id="ASSESSMENT-REPRO-001",
        experiment_id="EXPERIMENT-REPRO-001",
        title="Agent security boundary violation",
        result=ExperimentResult.FAIL,
        confidence="High",
        affected_properties=[
            "action_boundary_preservation",
        ],
        rationale="Unauthorized action observed.",
        evidence={
            "source": "controlled-test",
        },
    )


def test_reproduction_starts_pending() -> None:
    reproduction = ReproductionBuilder().create(
        reproduction_id="REPRO-001",
        finding=build_finding(),
        follow_up_experiment_id="EXPERIMENT-REPRO-FOLLOWUP-001",
    )

    assert reproduction.status == ReproductionStatus.PENDING
    assert reproduction.result == ExperimentResult.INCONCLUSIVE
    assert reproduction.finding_id == "FINDING-REPRO-001"
    assert (
        reproduction.source_experiment_id
        == "EXPERIMENT-REPRO-001"
    )


def test_failed_follow_up_reproduces_finding() -> None:
    builder = ReproductionBuilder()

    reproduction = builder.create(
        reproduction_id="REPRO-002",
        finding=build_finding(),
        follow_up_experiment_id="EXPERIMENT-REPRO-FOLLOWUP-002",
    )

    resolved = builder.resolve(
        reproduction=reproduction,
        result=ExperimentResult.FAIL,
        confidence="High",
        rationale="The same security boundary violation occurred.",
        evidence={
            "property": "action_boundary_preservation",
        },
    )

    assert resolved.status == ReproductionStatus.REPRODUCED
    assert resolved.result == ExperimentResult.FAIL
    assert resolved.confidence == "High"
    assert (
        resolved.evidence["follow_up_result"]["result"]
        == "FAIL"
    )


def test_passing_follow_up_does_not_reproduce_finding() -> None:
    builder = ReproductionBuilder()

    reproduction = builder.create(
        reproduction_id="REPRO-003",
        finding=build_finding(),
        follow_up_experiment_id="EXPERIMENT-REPRO-FOLLOWUP-003",
    )

    resolved = builder.resolve(
        reproduction=reproduction,
        result=ExperimentResult.PASS,
        confidence="High",
        rationale="The controlled follow-up remained within the boundary.",
    )

    assert resolved.status == ReproductionStatus.NOT_REPRODUCED
    assert resolved.result == ExperimentResult.PASS


def test_inconclusive_follow_up_remains_inconclusive() -> None:
    builder = ReproductionBuilder()

    reproduction = builder.create(
        reproduction_id="REPRO-004",
        finding=build_finding(),
        follow_up_experiment_id="EXPERIMENT-REPRO-FOLLOWUP-004",
    )

    resolved = builder.resolve(
        reproduction=reproduction,
        result=ExperimentResult.INCONCLUSIVE,
        confidence="Low",
        rationale="The system did not expose enough observable behavior.",
    )

    assert resolved.status == ReproductionStatus.INCONCLUSIVE
    assert resolved.result == ExperimentResult.INCONCLUSIVE
