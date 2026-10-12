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
from aivexa.targets.config import TargetConfig
from aivexa.targets.factory import build_model_target
from aivexa.evaluation.model_security_runner import (
    ModelSecurityRunner,
    ModelSecuritySuite,
)


VERSION = "0.7.0"


def _build_model_target(args):
    """
    Build a model target from CLI arguments using the provider factory.
    """
    config = TargetConfig(
        name=args.target_name or args.model,
        target_type=getattr(args, "target_type", "ollama"),
        endpoint=args.endpoint,
        model=args.model,
        api_key_env=getattr(args, "api_key_env", "") or "",
        timeout=getattr(args, "timeout", 120),
    )

    return build_model_target(config)


def _semantic_evaluator(args):
    """
    Build the optional semantic relevance evaluator when requested.
    """
    if not getattr(args, "semantic", False):
        return None

    from aivexa.evaluation.semantic_relevance import (
        EmbeddingClient,
        SemanticRelevanceEvaluator,
    )

    client = EmbeddingClient(
        endpoint=args.endpoint,
        model=args.embedding_model,
    )

    return SemanticRelevanceEvaluator(client)


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

    target_profile = target_sub.add_parser(
        "profile",
        help="Actively profile a model target's capabilities.",
    )
    target_profile.add_argument("--model", required=True)
    target_profile.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )
    target_profile.add_argument(
        "--target-type",
        default="ollama",
    )
    target_profile.add_argument(
        "--api-key-env",
        default="",
    )
    target_profile.add_argument(
        "--timeout",
        type=int,
        default=120,
    )
    target_profile.add_argument(
        "--target-name",
        default="",
    )
    target_profile.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    target_list = target_sub.add_parser(
        "list",
        help="List configured targets in the target registry.",
    )
    target_list.add_argument(
        "--config",
        default="aivexa-targets.json",
    )
    target_list.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    target_show = target_sub.add_parser(
        "show",
        help="Show one configured target.",
    )
    target_show.add_argument("name")
    target_show.add_argument(
        "--config",
        default="aivexa-targets.json",
    )

    target_add = target_sub.add_parser(
        "add",
        help="Add or update a target in the registry.",
    )
    target_add.add_argument("name")
    target_add.add_argument(
        "--type",
        required=True,
        dest="target_type",
        help="Target provider type (e.g. gemini, ollama).",
    )
    target_add.add_argument(
        "--endpoint",
        required=True,
    )
    target_add.add_argument("--model", default="")
    target_add.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable that holds the API key.",
    )
    target_add.add_argument(
        "--timeout",
        type=int,
        default=120,
    )
    target_add.add_argument(
        "--config",
        default="aivexa-targets.json",
    )

    target_remove = target_sub.add_parser(
        "remove",
        help="Remove a configured target.",
    )
    target_remove.add_argument("name")
    target_remove.add_argument(
        "--config",
        default="aivexa-targets.json",
    )

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
        help="Model endpoint.",
    )

    model_safety_run.add_argument(
        "--target-type",
        default="ollama",
        help=(
            "Provider type: ollama or openai-compatible."
        ),
    )

    model_safety_run.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable holding the API key.",
    )

    model_safety_run.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )

    model_safety_run.add_argument(
        "--target-name",
        default="",
        help="Human-readable target name for records.",
    )

    model_safety_run.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Enable the optional embedding-based semantic relevance "
            "refinement (requires an embedding endpoint)."
        ),
    )

    model_safety_run.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model for --semantic.",
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

    model_security_parser = subparsers.add_parser(
        "model-security",
        help="Run model security evaluations.",
    )

    model_security_sub = model_security_parser.add_subparsers(
        dest="model_security_command",
        required=True,
    )

    model_security_run = model_security_sub.add_parser(
        "run",
        help="Run controlled model security evaluations.",
    )

    model_security_run.add_argument(
        "--model",
        required=True,
    )

    model_security_run.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )

    model_security_run.add_argument(
        "--target-type",
        default="ollama",
    )

    model_security_run.add_argument(
        "--api-key-env",
        default="",
    )

    model_security_run.add_argument(
        "--timeout",
        type=int,
        default=120,
    )

    model_security_run.add_argument(
        "--target-name",
        default="",
    )

    model_security_run.add_argument(
        "--db",
        default="aivexa.db",
    )

    model_security_run.add_argument(
        "--case",
        action="append",
        dest="cases",
    )

    model_security_run.add_argument(
        "--temperature",
        type=float,
    )

    model_security_run.add_argument(
        "--num-predict",
        type=int,
    )

    model_security_run.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
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
        "--adaptive-cycles",
        type=int,
        default=0,
        help=(
            "Execute up to N adaptive follow-up experiments when a "
            "finding is produced."
        ),
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

    model_safety_report = report_sub.add_parser(
        "model-safety",
        help=(
            "Generate a report from a model-safety results JSON file."
        ),
    )
    model_safety_report.add_argument("results")
    model_safety_report.add_argument(
        "--output",
        default="reports/model_safety_report.md",
    )

    experiment_report = report_sub.add_parser(
        "experiment",
        help="Generate a report for any persisted experiment.",
    )
    experiment_report.add_argument("experiment_id")
    experiment_report.add_argument(
        "--db",
        default="aivexa.db",
    )
    experiment_report.add_argument(
        "--output",
    )

    research_parser = subparsers.add_parser(
        "research",
        help="Research, benchmarking, and corpus operations.",
    )
    research_sub = research_parser.add_subparsers(
        dest="research_command",
        required=True,
    )

    research_export = research_sub.add_parser(
        "export",
        help="Export the research dataset.",
    )
    research_export.add_argument("path")
    research_export.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
    )
    research_export.add_argument(
        "--db",
        default="aivexa.db",
    )

    research_metrics = research_sub.add_parser(
        "metrics",
        help="Print descriptive research metrics.",
    )
    research_metrics.add_argument(
        "--db",
        default="aivexa.db",
    )
    research_metrics.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_coverage = research_sub.add_parser(
        "coverage",
        help="Print research coverage dimensions.",
    )
    research_coverage.add_argument(
        "--db",
        default="aivexa.db",
    )
    research_coverage.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_reliability = research_sub.add_parser(
        "reliability",
        help="Measure evaluator reliability across repeated runs.",
    )
    research_reliability.add_argument(
        "--db",
        default="aivexa.db",
    )
    research_reliability.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_efficiency = research_sub.add_parser(
        "efficiency",
        help="Compare fixed and adaptive evaluation efficiency.",
    )
    research_efficiency.add_argument(
        "--db",
        default="aivexa.db",
    )
    research_efficiency.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_knowledge = research_sub.add_parser(
        "knowledge",
        help="Search the local knowledge corpus.",
    )
    research_knowledge.add_argument("query")
    research_knowledge.add_argument(
        "--corpus",
        default="knowledge-corpus.jsonl",
    )
    research_knowledge.add_argument("--cwe")
    research_knowledge.add_argument("--tag")
    research_knowledge.add_argument(
        "--limit",
        type=int,
        default=10,
    )
    research_knowledge.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_refresh = research_sub.add_parser(
        "refresh",
        help="Refresh the knowledge corpus from an advisory source.",
    )
    research_refresh.add_argument(
        "--source",
        choices=["nvd", "jsonl", "http"],
        default="nvd",
    )
    research_refresh.add_argument(
        "--corpus",
        default="knowledge-corpus.jsonl",
    )
    research_refresh.add_argument(
        "--file",
        help="Path to a JSONL file (source=jsonl).",
    )
    research_refresh.add_argument(
        "--url",
        help="URL to an HTTP JSON source (source=http).",
    )
    research_refresh.add_argument(
        "--api-key-env",
        default="NVD_API_KEY",
    )
    research_refresh.add_argument(
        "--since",
        help=(
            "Incremental refresh start (ISO-8601 last-modified)."
        ),
    )
    research_refresh.add_argument(
        "--max-pages",
        type=int,
        default=1,
    )
    research_refresh.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    research_refresh.add_argument(
        "--results-per-page",
        type=int,
        default=100,
    )
    research_refresh.add_argument(
        "--timeout",
        type=int,
        default=60,
    )
    research_refresh.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )

    research_status = research_sub.add_parser(
        "knowledge-status",
        help="Report corpus size and staleness.",
    )
    research_status.add_argument(
        "--corpus",
        default="knowledge-corpus.jsonl",
    )
    research_status.add_argument(
        "--max-age-days",
        type=int,
        default=30,
    )
    research_status.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
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

    target = _build_model_target(args)

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
        semantic_evaluator=_semantic_evaluator(args),
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


def _run_model_security(args) -> int:
    database = Database(args.db)

    target = _build_model_target(args)

    generation_options = _generation_options(args)

    cases = None

    if args.cases:
        available = {
            case.case_id: case
            for case in ModelSecuritySuite.default_cases()
        }

        unknown = [
            case_id
            for case_id in args.cases
            if case_id not in available
        ]

        if unknown:
            raise ValueError(
                "Unknown model security case(s): "
                + ", ".join(unknown)
                + ". Available cases: "
                + ", ".join(available)
            )

        cases = tuple(
            available[case_id] for case_id in args.cases
        )

    runner = ModelSecurityRunner(
        runner=ExperimentRunner(target),
        database=database,
    )

    result = runner.run(
        cases=cases,
        generation_options=generation_options,
    )

    payload = result.to_dict()

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return 0

    print("AIVEXA MODEL SECURITY EVALUATION")
    print("================================")
    print(f"TARGET: {payload['target']}")
    print()

    for execution in payload["executions"]:
        case = execution["case"]
        evaluation = execution["evaluation"]

        print(
            f"{case['case_id']}: "
            f"{evaluation['result']} "
            f"({evaluation['confidence']})"
        )

        if execution["finding"]:
            print(
                "  FINDING: "
                f"{execution['finding']['finding_id']}"
            )

    print()
    print("SUMMARY")
    print("-------")

    for key, value in payload["summary"].items():
        print(f"{key}: {value}")

    return 0


def _run_target_profile(args) -> int:
    from aivexa.targets.profiling import TargetProfiler

    target = _build_model_target(args)

    result = TargetProfiler(
        timeout=getattr(args, "timeout", 120),
    ).profile(target)

    payload = result.to_dict()

    if getattr(args, "json_output", False):
        print(json.dumps(payload, indent=2))
        return 0 if result.reachable else 1

    print("AIVEXA TARGET PROFILE")
    print("=====================")
    print(f"NAME: {result.name}")
    print(f"MODEL: {result.model}")
    print(f"ENDPOINT: {result.endpoint}")
    print(f"REACHABLE: {result.reachable}")

    if result.latency_ms is not None:
        print(f"LATENCY_MS: {result.latency_ms}")

    print("CAPABILITIES:")
    for capability, supported in result.capabilities.items():
        print(f"  {capability}: {supported}")

    for observation in result.observations:
        print(f"- {observation}")

    for error in result.errors:
        print(f"ERROR: {error}")

    return 0 if result.reachable else 1


def _run_research(args) -> int:
    if args.research_command == "export":
        from aivexa.research.dataset import ResearchDataset

        database = Database(args.db)
        dataset = ResearchDataset(database)

        if args.format == "jsonl":
            path = dataset.export_jsonl(args.path)
        else:
            path = dataset.export_json(args.path)

        print(f"EXPORTED: {path}")
        return 0

    if args.research_command == "metrics":
        from aivexa.research.metrics import (
            ResearchMetricsCalculator,
        )

        database = Database(args.db)
        metrics = ResearchMetricsCalculator(
            database
        ).calculate()
        payload = metrics.to_dict()

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA RESEARCH METRICS")
        print("=======================")
        print(f"experiments: {payload['total_experiments']}")
        print(f"targets: {payload['unique_targets']}")
        print(f"properties: {payload['unique_properties']}")
        print(f"evaluators: {payload['unique_evaluators']}")
        print("results:")
        for name, count in payload[
            "result_distribution"
        ]["counts"].items():
            print(f"  {name}: {count}")
        return 0

    if args.research_command == "coverage":
        from aivexa.research.coverage import (
            CoverageCalculator,
        )

        database = Database(args.db)
        coverage = CoverageCalculator(
            database
        ).calculate()
        payload = coverage.to_dict()

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA RESEARCH COVERAGE")
        print("========================")
        print(f"experiments: {payload['total_experiments']}")
        for name in (
            "target_coverage",
            "property_coverage",
            "evaluator_coverage",
            "intervention_coverage",
        ):
            dimension = payload[name]
            print(
                f"{name}: {dimension['covered']}/"
                f"{dimension['total']} "
                f"({dimension['proportion']:.2f})"
            )
        return 0

    if args.research_command == "reliability":
        from aivexa.research.reliability import (
            EvaluatorReliabilityCalculator,
        )

        database = Database(args.db)
        reliability = EvaluatorReliabilityCalculator(
            database
        ).calculate()
        payload = [item.to_dict() for item in reliability]

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA EVALUATOR RELIABILITY")
        print("============================")
        if not payload:
            print("No repeated evaluator groups were found.")
            return 0

        for item in payload:
            print(
                f"{item['evaluator']}: "
                f"agreement={item['agreement_rate']:.2f} "
                f"({item['consistent_groups']}/"
                f"{item['repeated_groups']} consistent)"
            )
        return 0

    if args.research_command == "efficiency":
        from aivexa.research.adaptive_efficiency import (
            AdaptiveEfficiencyAnalyzer,
        )

        database = Database(args.db)
        comparison = AdaptiveEfficiencyAnalyzer(
        ).compare_from_database(database)
        payload = comparison.to_dict()

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA ADAPTIVE EFFICIENCY")
        print("==========================")
        print(f"fixed experiments: {payload['fixed']['experiments']}")
        print(
            "fixed findings: "
            f"{payload['fixed']['confirmed_findings']}"
        )
        print(
            "adaptive experiments: "
            f"{payload['adaptive']['experiments']}"
        )
        print(
            "adaptive findings: "
            f"{payload['adaptive']['confirmed_findings']}"
        )
        print(payload["rationale"])
        return 0

    if args.research_command == "knowledge":
        from aivexa.knowledge.corpus import KnowledgeCorpus

        corpus = KnowledgeCorpus(args.corpus)
        entries = corpus.search(
            query=args.query,
            cwe=args.cwe,
            tag=args.tag,
            limit=args.limit,
        )

        payload = [entry.to_dict() for entry in entries]

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA KNOWLEDGE SEARCH")
        print("=======================")
        if not payload:
            print("No matching corpus entries.")
            return 0

        for entry in payload:
            print(
                f"- {entry['entry_id']}: {entry['title']} "
                f"[source={entry['source']}]"
            )
        return 0

    if args.research_command == "refresh":
        from aivexa.knowledge.corpus import KnowledgeCorpus
        from aivexa.knowledge.nvd import NvdCveSource
        from aivexa.knowledge.refresh import refresh_corpus
        from aivexa.knowledge.sources import (
            HttpCorpusSource,
            JsonlCorpusSource,
        )

        if args.source == "nvd":
            source = NvdCveSource(
                api_key_env=args.api_key_env,
                timeout=args.timeout,
            )
        elif args.source == "jsonl":
            if not args.file:
                raise ValueError(
                    "--file is required for source=jsonl"
                )
            source = JsonlCorpusSource("jsonl", args.file)
        else:
            if not args.url:
                raise ValueError(
                    "--url is required for source=http"
                )
            source = HttpCorpusSource(
                "http",
                args.url,
                timeout=args.timeout,
            )

        corpus = KnowledgeCorpus(args.corpus)

        report = refresh_corpus(
            corpus,
            source,
            max_pages=args.max_pages,
            start_index=args.start_index,
            results_per_page=args.results_per_page,
            last_mod_start=args.since,
        )

        payload = report.to_dict()

        if args.json_output:
            print(json.dumps(payload, indent=2))
        else:
            print("AIVEXA KNOWLEDGE REFRESH")
            print("========================")
            print(f"source: {report.source}")
            print(f"success: {report.success}")
            print(f"pages: {report.pages}")
            print(
                "entries_fetched: "
                f"{report.entries_fetched}"
            )
            print(f"added: {report.added}")
            print(f"updated: {report.updated}")
            print(f"unchanged: {report.unchanged}")
            print(
                "next_start_index: "
                f"{report.next_start_index}"
            )
            print(f"exhausted: {report.exhausted}")

            for error in report.errors:
                print(f"ERROR: {error}")

        return 0 if report.success else 1

    if args.research_command == "knowledge-status":
        from datetime import datetime, timezone

        from aivexa.knowledge.corpus import KnowledgeCorpus

        corpus = KnowledgeCorpus(args.corpus)
        entries = corpus.all()

        now = datetime.now(timezone.utc)
        stale = [
            entry
            for entry in entries
            if corpus.is_stale(
                entry,
                max_age_days=args.max_age_days,
                now=now,
            )
        ]

        retrieved = sorted(
            entry.retrieved_at
            for entry in entries
            if entry.retrieved_at
        )

        sources: dict[str, int] = {}

        for entry in entries:
            sources[entry.source] = (
                sources.get(entry.source, 0) + 1
            )

        payload = {
            "corpus": args.corpus,
            "entries": len(entries),
            "stale_entries": len(stale),
            "max_age_days": args.max_age_days,
            "newest_retrieved_at": (
                retrieved[-1] if retrieved else None
            ),
            "sources": sources,
        }

        if args.json_output:
            print(json.dumps(payload, indent=2))
            return 0

        print("AIVEXA KNOWLEDGE STATUS")
        print("=======================")
        print(f"corpus: {args.corpus}")
        print(f"entries: {len(entries)}")
        print(f"stale_entries: {len(stale)}")
        print(
            "newest_retrieved_at: "
            f"{payload['newest_retrieved_at']}"
        )
        for name, count in sorted(sources.items()):
            print(f"  {name}: {count}")

        return 0

    raise ValueError(
        f"Unsupported research command: {args.research_command}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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

            if args.target_command == "profile":
                return _run_target_profile(args)

            if args.target_command == "list":
                from aivexa.targets.config import (
                    TargetRegistry,
                )

                targets = TargetRegistry(
                    args.config
                ).list()

                if args.json_output:
                    print(
                        json.dumps(
                            [
                                config.to_dict()
                                for config in targets
                            ],
                            indent=2,
                        )
                    )
                    return 0

                if not targets:
                    print(
                        "No targets are configured in "
                        f"{args.config}."
                    )
                    return 0

                for config in targets:
                    print(
                        f"{config.name}: {config.target_type} "
                        f"({config.endpoint})"
                    )

                return 0

            if args.target_command == "show":
                from aivexa.targets.config import (
                    TargetRegistry,
                )

                config = TargetRegistry(
                    args.config
                ).get(args.name)

                print(
                    json.dumps(
                        config.to_dict(),
                        indent=2,
                    )
                )
                return 0

            if args.target_command == "add":
                from aivexa.targets.config import (
                    TargetConfig,
                    TargetRegistry,
                )

                registry = TargetRegistry(args.config)

                registry.save(
                    TargetConfig(
                        name=args.name,
                        target_type=args.target_type,
                        endpoint=args.endpoint,
                        model=args.model,
                        api_key_env=args.api_key_env,
                        timeout=args.timeout,
                    )
                )

                print(
                    f"SAVED: {args.name} -> {args.config}"
                )
                return 0

            if args.target_command == "remove":
                from aivexa.targets.config import (
                    TargetRegistry,
                )

                TargetRegistry(args.config).delete(
                    args.name
                )

                print(
                    f"REMOVED: {args.name} from "
                    f"{args.config}"
                )
                return 0

        if args.command == "adversarial":
            return _run_adversarial(args)

        if args.command == "model-safety":
            return _run_model_safety(args)

        if args.command == "model-security":
            return _run_model_security(args)

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
                max_adaptive_cycles=args.adaptive_cycles,
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

            adaptive_run = None

            if (
                args.adaptive_cycles > 0
                and result.finding is not None
            ):
                adaptive_run = runner.investigate(
                    source_experiment_id=args.experiment_id,
                    source_input=args.input,
                    checks=build_checks(args.checks),
                    context=_json_object(args.context),
                    authorization_context=_json_object(
                        args.authorization_context
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
                "adaptive": (
                    adaptive_run.to_dict()
                    if adaptive_run is not None
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

                if payload["adaptive"]:
                    print(
                        "Adaptive: "
                        f"{payload['adaptive']['stopping_reason']} "
                        f"({len(payload['adaptive']['cycles'])} "
                        "cycle(s))"
                    )

                    for cycle in payload["adaptive"]["cycles"]:
                        print(
                            f"  cycle {cycle['cycle_number']}: "
                            f"{cycle['result']} "
                            f"[{cycle['strategy']}]"
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

            if args.report_command == "model-safety":
                from aivexa.reporting.model_safety_report import (
                    generate_model_safety_report,
                )

                report_path = generate_model_safety_report(
                    results_file=args.results,
                    report_file=args.output,
                )

                print(f"REPORT: {report_path}")
                return 0

            if args.report_command == "experiment":
                from aivexa.reporting.report import (
                    ReportGenerator,
                )

                database = Database(args.db)

                report = ReportGenerator(
                    database
                ).generate(
                    args.experiment_id,
                    output_path=args.output,
                )

                if not args.output:
                    print(report)

                return 0

        if args.command == "research":
            return _run_research(args)

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
