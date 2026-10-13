from __future__ import annotations

import argparse
import json
from pathlib import Path

from aivexa.evaluation.model_safety import ModelSafetySuite
from aivexa.evaluation.safety_boundary_engine import SafetyBoundaryEngine
from aivexa.execution import ExecutionPolicy
from aivexa.targets.ollama import OllamaTarget
from aivexa.reporting.model_safety_report import generate_model_safety_report


def execute_prompt(
    target: OllamaTarget,
    prompt: str,
    generation_options: dict | None = None,
) -> dict:
    """
    Execute one prompt.

    Retries and timeout handling live in the target's shared
    ``ExecutionPolicy``, so there is no second retry loop here. The
    attempt count is recorded from the target's execution telemetry.
    """
    try:
        output = target.generate(
            prompt,
            options=generation_options,
        )
        telemetry = getattr(target, "last_telemetry", None)
        return {
            "status": "COMPLETED",
            "output": output,
            "error": None,
            "attempts": (
                telemetry.attempts
                if telemetry is not None
                else 1
            ),
        }
    except Exception as exc:
        telemetry = getattr(target, "last_telemetry", None)
        return {
            "status": "ERROR",
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": (
                telemetry.attempts
                if telemetry is not None
                else 0
            ),
        }


def run_case(
    target: OllamaTarget,
    case,
    evaluator: SafetyBoundaryEngine,
    generation_options: dict | None = None,
) -> dict:
    baseline_execution = execute_prompt(
        target,
        case.baseline_prompt,
        generation_options=generation_options,
    )

    if baseline_execution["status"] == "ERROR":
        return {
            "case_id": case.case_id,
            "test_type": case.test_type.value,
            "objective": case.objective,
            "hypothesis": case.hypothesis,
            "expected_property": case.expected_property,
            "baseline_prompt": case.baseline_prompt,
            "baseline_output": None,
            "baseline_execution": baseline_execution,
            "variants": [],
            "case_status": "EXECUTION_ERROR",
        }

    baseline = baseline_execution["output"]
    variants = []

    for index, variant_prompt in enumerate(case.variants, start=1):
        execution = execute_prompt(
            target,
            variant_prompt,
            generation_options=generation_options,
        )

        if execution["status"] == "ERROR":
            variants.append(
                {
                    "variant_number": index,
                    "prompt": variant_prompt,
                    "output": None,
                    "execution": execution,
                    "result": "INCONCLUSIVE",
                    "confidence": "Low",
                    "rationale": (
                        "The model execution failed before "
                        "the safety property could be evaluated."
                    ),
                    "evidence": {
                        "execution_error": execution["error"],
                    },
                }
            )
            continue

        output = execution["output"]

        try:
            assessment = evaluator.evaluate_responses(
                baseline_response=baseline,
                variant_responses=[output],
                topic_keywords=tuple(
                    getattr(case, "topic_keywords", ()) or ()
                ),
            )

            variants.append(
                {
                    "variant_number": index,
                    "prompt": variant_prompt,
                    "output": output,
                    "execution": execution,
                    "result": assessment.result.value,
                    "confidence": assessment.confidence,
                    "rationale": assessment.rationale,
                    "evidence": assessment.evidence,
                }
            )

        except Exception as exc:
            variants.append(
                {
                    "variant_number": index,
                    "prompt": variant_prompt,
                    "output": output,
                    "execution": execution,
                    "result": "INCONCLUSIVE",
                    "confidence": "Low",
                    "rationale": (
                        "The model response was captured, but "
                        "evaluation failed."
                    ),
                    "evidence": {
                        "evaluation_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    },
                }
            )

    completed = sum(
        1
        for variant in variants
        if variant["execution"]["status"] == "COMPLETED"
    )

    errors = len(variants) - completed

    if errors == len(variants):
        case_status = "EXECUTION_ERROR"
    elif errors > 0:
        case_status = "PARTIAL_EXECUTION"
    else:
        case_status = "COMPLETED"

    return {
        "case_id": case.case_id,
        "test_type": case.test_type.value,
        "objective": case.objective,
        "hypothesis": case.hypothesis,
        "expected_property": case.expected_property,
        "baseline_prompt": case.baseline_prompt,
        "baseline_output": baseline,
        "baseline_execution": baseline_execution,
        "variants": variants,
        "case_status": case_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AIVEXA model safety evaluation cases."
    )

    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
    )

    parser.add_argument(
        "--output",
        default="model_safety_results.json",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-attempt socket timeout in seconds.",
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Maximum tokens to generate per request.",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "Maximum attempts per request (transient failures only)."
        ),
    )

    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Initial retry backoff in seconds.",
    )

    args = parser.parse_args()

    policy = ExecutionPolicy(
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff,
    )

    target = OllamaTarget(
        endpoint=args.endpoint,
        model=args.model,
        policy=policy,
    )

    generation_options = {}

    if args.num_predict is not None:
        generation_options["num_predict"] = args.num_predict

    evaluator = SafetyBoundaryEngine()
    cases = ModelSafetySuite.default_cases()

    results = []

    print(f"TARGET: {args.model}")
    print(f"CASES: {len(cases)}")
    print()

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] "
            f"{case.case_id} "
            f"({case.test_type.value})"
        )

        result = run_case(
            target=target,
            case=case,
            evaluator=evaluator,
            generation_options=generation_options,
        )

        results.append(result)

        variant_results = [
            variant["result"]
            for variant in result["variants"]
        ]

        print(
            "  RESULTS: "
            + (
                ", ".join(variant_results)
                if variant_results
                else "NO_VARIANTS_COMPLETED"
            )
        )

        if result["case_status"] != "COMPLETED":
            print(
                f"  STATUS: {result['case_status']}"
            )

    output = {
        "target": {
            "endpoint": args.endpoint,
            "model": args.model,
        },
        "suite": {
            "name": "AIVEXA Model Safety Suite",
            "case_count": len(cases),
        },
        "results": results,
    }

    Path(args.output).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    report_path = generate_model_safety_report(
        results_file=args.output,
        report_file=(
            Path("reports")
            / f"{args.model.replace('/', '_').replace(':', '_')}_model_safety_report.md"
        ),
    )

    print()
    print(f"FULL REPORT: {report_path}")

    counts: dict[str, int] = {}

    for result in results:
        for variant in result["variants"]:
            value = variant["result"]
            counts[value] = counts.get(value, 0) + 1

    print()
    print("SUMMARY")
    print("-------")

    for result_name in (
        "PASS",
        "FAIL",
        "ANOMALY",
        "INCONCLUSIVE",
        "NOT_APPLICABLE",
    ):
        print(
            f"{result_name}: "
            f"{counts.get(result_name, 0)}"
        )

    case_statuses: dict[str, int] = {}

    for result in results:
        status = result["case_status"]
        case_statuses[status] = case_statuses.get(status, 0) + 1

    print()
    print("CASE EXECUTION")
    print("--------------")

    for status, count in case_statuses.items():
        print(f"{status}: {count}")

    print()
    print(f"EVIDENCE: {args.output}")


if __name__ == "__main__":
    main()
