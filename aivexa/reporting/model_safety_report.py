from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_NAMES = (
    "PASS",
    "FAIL",
    "ANOMALY",
    "INCONCLUSIVE",
    "NOT_APPLICABLE",
)


def _count_results(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in RESULT_NAMES}

    for case in results:
        for variant in case.get("variants", []):
            result = variant.get("result", "INCONCLUSIVE")
            counts[result] = counts.get(result, 0) + 1

    return counts


def _execution_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "COMPLETED": 0,
        "PARTIAL_EXECUTION": 0,
        "EXECUTION_ERROR": 0,
    }

    for case in results:
        status = case.get("case_status", "EXECUTION_ERROR")
        counts[status] = counts.get(status, 0) + 1

    return counts


def _final_assessment(counts: dict[str, int]) -> str:
    if counts.get("FAIL", 0) > 0:
        return (
            "Confirmed safety-boundary failures were observed within "
            "the tested scope. These results require investigation, "
            "reproduction, and impact assessment."
        )

    if counts.get("ANOMALY", 0) > 0:
        return (
            "No confirmed safety-boundary failure was established, "
            "but anomalous behavior was observed and requires further "
            "investigation."
        )

    if counts.get("INCONCLUSIVE", 0) > 0:
        return (
            "No confirmed safety-boundary failure was established, "
            "but some experiments were inconclusive and the assessment "
            "should not be treated as complete."
        )

    return (
        "No confirmed safety-boundary failure was observed within "
        "the tested scope."
    )


def _interpretation(result: str) -> str:
    interpretations = {
        "PASS": (
            "The observed behavior matched the expected safety "
            "property for this experiment."
        ),
        "FAIL": (
            "The evaluator identified evidence that the expected "
            "safety boundary was violated."
        ),
        "ANOMALY": (
            "The observed behavior differed from the expected "
            "boundary or showed unexpected behavior. An anomaly "
            "is not automatically a vulnerability."
        ),
        "INCONCLUSIVE": (
            "AIVEXA did not have sufficient evidence to confidently "
            "classify the behavior."
        ),
        "NOT_APPLICABLE": (
            "The experiment could not meaningfully evaluate the "
            "property for this target or context."
        ),
    }

    return interpretations.get(
        result,
        "The result could not be interpreted.",
    )


def generate_model_safety_report(
    results_file: str | Path = "model_safety_results.json",
    report_file: str | Path = "reports/model_safety_report.md",
) -> Path:
    results_path = Path(results_file)
    report_path = Path(report_file)

    payload = json.loads(
        results_path.read_text(encoding="utf-8")
    )

    target = payload.get("target", {})
    suite = payload.get("suite", {})
    results = payload.get("results", [])

    if not isinstance(target, dict):
        target = {"name": str(target)}

    if not isinstance(suite, dict):
        suite = {}

    if not isinstance(results, list):
        results = []

    counts = _count_results(results)
    execution_counts = _execution_counts(results)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = []

    lines.extend(
        [
            "# AIVEXA Model Safety Assessment",
            "",
            "> Detailed evidence-backed assessment of AI model behavior "
            "under controlled safety experiments.",
            "",
            "---",
            "",
            "## Assessment Overview",
            "",
            f"**Generated:** "
            f"{datetime.now(timezone.utc).isoformat()}",
            "",
            f"**Assessment:** "
            f"{suite.get('name', 'AIVEXA Model Safety Suite')}",
            "",
            f"**Cases:** "
            f"{suite.get('case_count', len(results))}",
            "",
        ]
    )

    lines.append("### Target")
    lines.append("")

    if target:
        for key, value in target.items():
            label = key.replace("_", " ").title()
            lines.append(f"**{label}:** `{value}`")
            lines.append("")
    else:
        lines.append("Target information was not recorded.")
        lines.append("")

    lines.extend(
        [
            "## Executive Summary",
            "",
            _final_assessment(counts),
            "",
            "This assessment describes only the behavior observed "
            "during the executed experiments. It does not establish "
            "that the target is universally safe or secure.",
            "",
            "## Result Summary",
            "",
            "| Result | Count | Meaning |",
            "|---|---:|---|",
        ]
    )

    for result_name in RESULT_NAMES:
        lines.append(
            f"| {result_name} | "
            f"{counts.get(result_name, 0)} | "
            f"{_interpretation(result_name)} |"
        )

    lines.extend(
        [
            "",
            "## Execution Summary",
            "",
            "| Status | Cases |",
            "|---|---:|",
        ]
    )

    for status in (
        "COMPLETED",
        "PARTIAL_EXECUTION",
        "EXECUTION_ERROR",
    ):
        lines.append(
            f"| {status} | {execution_counts.get(status, 0)} |"
        )

    lines.extend(
        [
            "",
            "## What AIVEXA Tested",
            "",
        ]
    )

    if not results:
        lines.extend(
            [
                "No test cases were recorded.",
                "",
            ]
        )

    for case_number, case in enumerate(results, start=1):
        case_id = case.get(
            "case_id",
            f"CASE-{case_number}",
        )
        test_type = case.get(
            "test_type",
            "UNKNOWN",
        )
        objective = case.get(
            "objective",
            "",
        )
        hypothesis = case.get(
            "hypothesis",
            "",
        )
        expected_property = case.get(
            "expected_property",
            "",
        )
        case_status = case.get(
            "case_status",
            "UNKNOWN",
        )

        lines.extend(
            [
                f"## {case_number}. `{case_id}`",
                "",
                f"**Test type:** `{test_type}`",
                "",
                f"**Execution status:** `{case_status}`",
                "",
            ]
        )

        if objective:
            lines.extend(
                [
                    "### Objective",
                    "",
                    objective,
                    "",
                ]
            )

        if hypothesis:
            lines.extend(
                [
                    "### Hypothesis",
                    "",
                    hypothesis,
                    "",
                ]
            )

        if expected_property:
            lines.extend(
                [
                    "### Expected Property",
                    "",
                    expected_property,
                    "",
                ]
            )

        baseline_prompt = case.get(
            "baseline_prompt",
        )

        if baseline_prompt:
            lines.extend(
                [
                    "### Baseline Test",
                    "",
                    "```text",
                    str(baseline_prompt),
                    "```",
                    "",
                ]
            )

        baseline_output = case.get(
            "baseline_output",
        )

        if baseline_output is not None:
            lines.extend(
                [
                    "### Baseline Model Response",
                    "",
                    "```text",
                    str(baseline_output),
                    "```",
                    "",
                ]
            )

        variants = case.get("variants", [])

        if not variants:
            lines.extend(
                [
                    "### Variants",
                    "",
                    "No variants were recorded.",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                "### Controlled Experiments",
                "",
            ]
        )

        for variant_number, variant in enumerate(
            variants,
            start=1,
        ):
            result = variant.get(
                "result",
                "INCONCLUSIVE",
            )
            confidence = variant.get(
                "confidence",
                "UNKNOWN",
            )
            prompt = variant.get(
                "prompt",
                "",
            )
            output = variant.get(
                "output",
            )
            rationale = variant.get(
                "rationale",
                "",
            )
            evidence = variant.get(
                "evidence",
                {},
            )
            execution = variant.get(
                "execution",
                {},
            )

            lines.extend(
                [
                    f"#### Experiment {variant_number}",
                    "",
                    f"**Result:** `{result}`",
                    "",
                    f"**Confidence:** `{confidence}`",
                    "",
                    f"**Execution:** "
                    f"`{execution.get('status', 'UNKNOWN')}`",
                    "",
                ]
            )

            if prompt:
                lines.extend(
                    [
                        "**Variant Input**",
                        "",
                        "```text",
                        str(prompt),
                        "```",
                        "",
                    ]
                )

            if output is not None:
                lines.extend(
                    [
                        "**Model Response**",
                        "",
                        "```text",
                        str(output),
                        "```",
                        "",
                    ]
                )

            if rationale:
                lines.extend(
                    [
                        "**Evaluation Rationale**",
                        "",
                        rationale,
                        "",
                    ]
                )

            if evidence:
                lines.extend(
                    [
                        "**Evidence**",
                        "",
                        "```json",
                        json.dumps(
                            evidence,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        ),
                        "```",
                        "",
                    ]
                )

            if execution.get("error"):
                lines.extend(
                    [
                        "**Execution Error**",
                        "",
                        "```text",
                        str(execution["error"]),
                        "```",
                        "",
                    ]
                )

    anomalies = []

    inconclusive = []

    failures = []

    for case in results:
        for variant in case.get("variants", []):
            result = variant.get(
                "result",
                "INCONCLUSIVE",
            )

            item = {
                "case_id": case.get("case_id"),
                "test_type": case.get("test_type"),
                "variant_number": variant.get(
                    "variant_number"
                ),
                "rationale": variant.get(
                    "rationale",
                    "",
                ),
            }

            if result == "ANOMALY":
                anomalies.append(item)
            elif result == "INCONCLUSIVE":
                inconclusive.append(item)
            elif result == "FAIL":
                failures.append(item)

    lines.extend(
        [
            "## Findings Requiring Attention",
            "",
        ]
    )

    if failures:
        lines.append("### Confirmed Failures")
        lines.append("")

        for item in failures:
            lines.append(
                f"- `{item['case_id']}` "
                f"variant {item['variant_number']}: "
                f"{item['rationale']}"
            )

        lines.append("")

    if anomalies:
        lines.append("### Anomalies")
        lines.append("")

        for item in anomalies:
            lines.append(
                f"- `{item['case_id']}` "
                f"variant {item['variant_number']}: "
                f"{item['rationale']}"
            )

        lines.append("")

    if inconclusive:
        lines.append("### Inconclusive Experiments")
        lines.append("")

        for item in inconclusive:
            lines.append(
                f"- `{item['case_id']}` "
                f"variant {item['variant_number']}: "
                f"{item['rationale']}"
            )

        lines.append("")

    if not failures and not anomalies and not inconclusive:
        lines.extend(
            [
                "No failures, anomalies, or inconclusive experiments "
                "were recorded.",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommended Follow-Up",
            "",
        ]
    )

    if failures:
        lines.extend(
            [
                "1. Reproduce each confirmed failure.",
                "2. Test controlled variations of the triggering input.",
                "3. Determine whether the behavior violates a meaningful "
                "security or safety property.",
                "4. Capture reproducible evidence before treating it "
                "as a finding.",
                "",
            ]
        )

    if anomalies:
        lines.extend(
            [
                "1. Re-run each anomalous experiment.",
                "2. Test paraphrases of the triggering input.",
                "3. Test controlled multi-turn variations.",
                "4. Compare the anomalous behavior against the baseline.",
                "5. Determine whether the anomaly is reproducible.",
                "",
            ]
        )

    if inconclusive:
        lines.extend(
            [
                "1. Re-run incomplete or failed experiments.",
                "2. Capture complete baseline and variant responses.",
                "3. Investigate execution and evaluator errors.",
                "",
            ]
        )

    if not failures and not anomalies and not inconclusive:
        lines.extend(
            [
                "Expand the assessment with additional safety properties "
                "and adversarial variations.",
                "",
            ]
        )

    lines.extend(
        [
            "## Coverage and Limitations",
            "",
            f"AIVEXA executed **{len(results)}** safety cases "
            f"containing **{sum(len(c.get('variants', [])) for c in results)}** "
            "controlled variants.",
            "",
            "The absence of a confirmed failure does not mean that "
            "the model is safe against every possible prompt, attack, "
            "context, application, or deployment configuration.",
            "",
            "This report is an evidence-backed interpretation of the "
            "specific experiments performed by AIVEXA.",
            "",
            "## Assessment Conclusion",
            "",
            _final_assessment(counts),
            "",
            "---",
            "",
            "Generated by **AIVEXA**.",
            "",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path
