from __future__ import annotations

import argparse
import json
from pathlib import Path

from aivexa.evaluation.model_safety import ModelSafetySuite
from aivexa.evaluation.safety_boundary_engine import SafetyBoundaryEngine
from aivexa.targets.ollama import OllamaTarget


def execute_prompt(
    target: OllamaTarget,
    prompt: str,
) -> dict:
    try:
        output = target.generate(prompt)
        return {
            "status": "COMPLETED",
            "output": output,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_case(
    target: OllamaTarget,
    case,
    evaluator: SafetyBoundaryEngine,
) -> dict:
    baseline_execution = execute_prompt(
        target,
        case.baseline_prompt,
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

    args = parser.parse_args()

    target = OllamaTarget(
        endpoint=args.endpoint,
        model=args.model,
    )

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
