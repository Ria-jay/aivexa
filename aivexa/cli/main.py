from __future__ import annotations

import argparse
import json
import sys

from aivexa.evaluation.authorized_research import (
    AuthorizedResearchRunner,
)
from aivexa.evaluation.security_checks import build_checks
from aivexa.reporting.research_report import (
    ResearchReportGenerator,
)
from aivexa.storage.database import Database
from aivexa.experiments.runner import ExperimentRunner
from aivexa.targets.ollama import OllamaTarget
from aivexa.evaluation.adversarial import (
    AdversarialCaseLibrary,
    TargetCapability,
)
from aivexa.evaluation.adversarial_profile import (
    AdversarialTargetProfile,
)
from aivexa.evaluation.adversarial_workflow import (
    AdversarialWorkflow,
)
from aivexa.evaluation.model_safety import (
    ModelSafetySuite,
)
from aivexa.evaluation.model_safety_runner import (
    ModelSafetyRunner,
)

from aivexa.targets.profile import TargetProfile
from aivexa.targets.profile_http import (
    ProfileHTTPApplicationTarget,
)


VERSION = "0.4.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aivexa",
        description=(
            "AIVEXA AI safety and security evaluation platform."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=VERSION,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    target_parser = subparsers.add_parser(
        "target",
        help="Target profile operations.",
    )
    target_sub = target_parser.add_subparsers(
        dest="target_command",
        required=True,
    )

    validate = target_sub.add_parser(
        "validate",
        help="Validate an authorized target profile.",
    )
    validate.add_argument("profile")

    adversarial_parser = subparsers.add_parser(
        "adversarial",
        help="Run property-driven adversarial AI evaluation.",
    )

    adversarial_sub = adversarial_parser.add_subparsers(
        dest="adversarial_command",
        required=True,
    )

    adversarial_run = adversarial_sub.add_parser(
        "run",
        help="Run adversarial evaluations.",
    )

    adversarial_run.add_argument(
        "--model",
        required=True,
    )

    adversarial_run.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )

    adversarial_run.add_argument(
        "--db",
        default="aivexa.db",
    )

    adversarial_run.add_argument(
        "--capability",
        action="append",
        required=True,
        choices=[
            capability.value
            for capability in TargetCapability
        ],
    )

    adversarial_run.add_argument(
        "--authorization-note",
        required=True,
    )

    adversarial_run.add_argument(
        "--temperature",
        type=float,
    )

    adversarial_run.add_argument(
        "--num-predict",
        type=int,
    )

    adversarial_run.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    model_safety_parser = subparsers.add_parser(
        "model-safety",
        help="Run model safety evaluations.",
    )

    model_safety_sub = model_safety_parser.add_subparsers(
        dest="model_safety_command",
        required=True,
    )

    model_safety_run = model_safety_sub.add_parser(
        "run",
        help="Run controlled model safety evaluations.",
    )

    model_safety_run.add_argument(
        "--model",
        required=True,
        help="Ollama model name.",
    )

    model_safety_run.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
        help="Ollama endpoint.",
    )

    model_safety_run.add_argument(
        "--db",
        default="aivexa.db",
        help="AIVEXA database path.",
    )

    model_safety_run.add_argument(
        "--case",
        action="append",
        dest="cases",
        help=(
            "Run only the named safety case. "
            "Can be supplied multiple times."
        ),
    )

    model_safety_run.add_argument(
        "--temperature",
        type=float,
        help="Model generation temperature.",
    )

    model_safety_run.add_argument(
        "--num-predict",
        type=int,
        help="Maximum number of tokens to predict.",
    )

    model_safety_run.add_argument(
        "--max-adaptive-cycles",
        type=int,
        default=0,
        help="Maximum bounded adaptive follow-up cycles.",
    )

    model_safety_run.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive follow-up evaluation.",
    )

    model_safety_run.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON.",
    )

    assess_parser = subparsers.add_parser(
        "assess",
        help="Run an authorized AI application assessment.",
    )
    assess_sub = assess_parser.add_subparsers(
        dest="assess_command",
        required=True,
    )

    run = assess_sub.add_parser(
        "run",
        help="Execute a controlled assessment.",
    )
    run.add_argument("profile")
    run.add_argument(
        "--input",
        required=True,
    )
    run.add_argument(
        "--db",
        default="aivexa.db",
    )
    run.add_argument(
        "--experiment-id",
        required=True,
    )
    run.add_argument(
        "--check",
        action="append",
        dest="checks",
        help="Run only the named security check.",
    )
    run.add_argument(
        "--authorization-context",
        default="{}",
    )
    run.add_argument(
        "--context",
        default="{}",
    )
    run.add_argument(
        "--baseline-output",
    )
    run.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate researcher-facing reports.",
    )
    report_sub = report_parser.add_subparsers(
        dest="report_command",
        required=True,
    )

    finding = report_sub.add_parser(
        "finding",
        help="Generate a report for a finding.",
    )
    finding.add_argument("finding_id")
    finding.add_argument(
        "--db",
        default="aivexa.db",
    )
    finding.add_argument(
        "--output",
    )

    return parser


def _json_object(value: str) -> dict:
    parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected a JSON object."
        )

    return parsed


def _generation_options(args) -> dict:
    generation_options = {}

    if args.temperature is not None:
        generation_options["temperature"] = (
            args.temperature
        )

    if args.num_predict is not None:
        generation_options["num_predict"] = (
            args.num_predict
        )

    return generation_options


def _run_adversarial(args) -> int:
    database = Database(args.db)

    target = OllamaTarget(
        endpoint=args.endpoint,
        model=args.model,
    )

    capabilities = frozenset(
        TargetCapability(value)
        for value in args.capability
    )

    profile = AdversarialTargetProfile(
        target_name=args.model,
        capabilities=capabilities,
        authorized=True,
        authorization_note=args.authorization_note,
    )

    generation_options = _generation_options(args)

    workflow = AdversarialWorkflow(
        runner=ExperimentRunner(target),
        database=database,
    )

    result = workflow.run(
        profile=profile,
        generation_options=generation_options,
    )

    payload = result.to_dict()

    if args.json_output:
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return 0

    print("AIVEXA ADVERSARIAL EVALUATION")
    print("=============================")
    print(
        f"TARGET: {profile.target_name}"
    )
    print(
        "CAPABILITIES: "
        + ", ".join(
            capability.value
            for capability in sorted(
                profile.capabilities,
                key=lambda item: item.value,
            )
        )
    )
    print()

    for item in payload["results"]:
        evaluation = item["evaluation"]

        print(
            f"{item['case']['case_id']}: "
            f"{evaluation['result']} "
            f"({evaluation['confidence']})"
        )

        print(
            f"  PROPERTY: "
            f"{evaluation['property_name']}"
        )

        if item["finding"]:
            print(
                f"  FINDING: "
                f"{item['finding']['finding_id']}"
            )

    if payload["skipped_cases"]:
        print()
        print("SKIPPED — NOT APPLICABLE")
        print("-----------------------")

        for case_id in payload["skipped_cases"]:
            print(
                f"- {case_id}"
            )

    print()
    print("SUMMARY")
    print("-------")

    for name, count in sorted(
        payload["summary"]["result_counts"].items()
    ):
        print(
            f"{name}: {count}"
        )

    print(
        f"Executed: "
        f"{payload['summary']['executed']}"
    )

    print(
        f"Skipped: "
        f"{payload['summary']['skipped']}"
    )

    print(
        f"Findings: "
        f"{payload['summary']['finding_count']}"
    )

    return 0


def _run_model_safety(args) -> int:
    database = Database(args.db)

    target = OllamaTarget(
        endpoint=args.endpoint,
        model=args.model,
    )

    generation_options = _generation_options(args)

    cases = None

    if args.cases:
        available_cases = {
            case.case_id: case
            for case in ModelSafetySuite.default_cases()
        }

        unknown_cases = [
            case_id
            for case_id in args.cases
            if case_id not in available_cases
        ]

        if unknown_cases:
            raise ValueError(
                "Unknown model safety case(s): "
                + ", ".join(unknown_cases)
                + ". Available cases: "
                + ", ".join(
                    available_cases
                )
            )

        cases = tuple(
            available_cases[case_id]
            for case_id in args.cases
        )

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(target),
        database=database,
        max_adaptive_cycles=args.max_adaptive_cycles,
    )

    result = runner.run(
        cases=cases,
        generation_options=generation_options,
        adaptive=not args.no_adaptive,
    )

    payload = result.to_dict()

    if args.json_output:
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return 0

    print("AIVEXA MODEL SAFETY EVALUATION")
    print("==============================")
    print(
        f"TARGET: {payload['target']}"
    )
    print()

    for execution in payload["executions"]:
        case_result = execution["case_result"]
        evaluation = execution["evaluation"]

        print(
            f"{case_result['case']['case_id']}: "
            f"{evaluation['result']} "
            f"({evaluation['confidence']})"
        )

        print(
            f"  PROPERTY: "
            f"{evaluation['property_name']}"
        )

        if execution["finding"]:
            print(
                f"  FINDING: "
                f"{execution['finding']['finding_id']}"
            )

        if execution["adaptive_run"] is not None:
            print(
                "  ADAPTIVE: executed"
            )

        print()

    print("SUMMARY")
    print("-------")

    for key, value in payload["summary"].items():
        print(
            f"{key}: {value}"
        )

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "target":
            if args.target_command == "validate":
                profile = TargetProfile.load(
                    args.profile
                )

                if not profile.authorized:
                    raise PermissionError(
                        "Profile is valid but not authorized."
                    )

                print(
                    f"VALID: {profile.name}"
                )
                print(
                    f"Endpoint: {profile.endpoint}"
                )
                print(
                    f"Authorization: {profile.authorization_note}"
                )
                return 0

        if args.command == "adversarial":
            return _run_adversarial(args)

        if args.command == "model-safety":
            return _run_model_safety(args)

        if args.command == "assess":
            profile = TargetProfile.load(
                args.profile
            )

            if not profile.authorized:
                raise PermissionError(
                    "Assessment refused: target profile is not "
                    "marked authorized."
                )

            target = ProfileHTTPApplicationTarget(
                profile
            )

            database = Database(args.db)

            runner = AuthorizedResearchRunner(
                target=target,
                database=database,
            )

            result = runner.assess(
                experiment_id=args.experiment_id,
                user_input=args.input,
                checks=build_checks(args.checks),
                authorization_context=_json_object(
                    args.authorization_context
                ),
                context=_json_object(
                    args.context
                ),
                baseline_output=args.baseline_output,
            )

            payload = {
                "assessment_id": (
                    result.assessment.assessment_id
                ),
                "experiment_id": (
                    result.assessment.experiment_id
                ),
                "result": (
                    result.assessment.result.value
                ),
                "confidence": result.assessment.confidence,
                "rationale": result.assessment.rationale,
                "finding_id": (
                    result.finding.finding_id
                    if result.finding
                    else None
                ),
                "follow_up_action": (
                    result.follow_up.action
                    if result.follow_up
                    else None
                ),
                "follow_up_experiment_id": (
                    result.investigation.experiment.experiment_id
                    if result.investigation
                    else None
                ),
            }

            if args.json_output:
                print(
                    json.dumps(
                        payload,
                        indent=2,
                    )
                )
            else:
                print(
                    f"Assessment: "
                    f"{payload['assessment_id']}"
                )
                print(
                    f"Result: {payload['result']}"
                )
                print(
                    f"Confidence: {payload['confidence']}"
                )
                print(
                    f"Rationale: {payload['rationale']}"
                )

                if payload["finding_id"]:
                    print(
                        f"Finding: {payload['finding_id']}"
                    )

                if payload["follow_up_experiment_id"]:
                    print(
                        "Follow-up: "
                        f"{payload['follow_up_experiment_id']}"
                    )

            return 0

        if args.command == "report":
            if args.report_command == "finding":
                database = Database(args.db)

                report = ResearchReportGenerator(
                    database
                ).finding_report(
                    args.finding_id
                )

                if args.output:
                    with open(
                        args.output,
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(report)
                else:
                    print(report)

                return 0

        parser.error("Unsupported command.")
        return 2

    except Exception as exc:
        print(
            f"AIVEXA ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
