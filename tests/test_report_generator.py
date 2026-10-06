from aivexa.reporting.report import ReportGenerator
from aivexa.storage.database import Database


def test_report_generator_reads_persisted_experiment(tmp_path):
    database = Database(
        str(tmp_path / "report.db")
    )

    database.save_experiment(
        {
            "experiment_id": "REPORT-001",
            "target": "test-target",
            "objective": "Test reporting.",
            "hypothesis": "Evidence remains traceable.",
            "intervention": "CONTROLLED_TEST",
            "context": {
                "phase": 4,
            },
            "created_at": "2026-01-01T00:00:00+00:00",
            "input_data": "hello",
            "output_data": "world",
            "result": "PASS",
            "rationale": "Boundary preserved.",
            "confidence": "High",
            "observations": [
                "Observed expected behavior."
            ],
            "property_name": "test_property",
            "property_expectation": "Remain safe.",
            "comparison_group": None,
            "evaluator": "test",
        }
    )

    report = ReportGenerator(database).generate(
        "REPORT-001"
    )

    assert "# AIVEXA Security Assessment Report" in report
    assert "REPORT-001" in report
    assert "Boundary preserved." in report
    assert "Evidence remains traceable." in report
