# AIVEXA Architecture

AIVEXA treats AI evaluation as an investigation. The fundamental unit
is the **Experiment**, and every conclusion is backed by evidence.

## Layer overview

```
targets/          Provider adapters and target configuration/profiling
experiments/      Experiment execution (model + application)
evidence/         Observable interaction and evidence records
evaluation/       Evaluators and assessment engines
comparison/       Deterministic behavioural comparison
storage/          SQLite persistence and schema migration
research/         Datasets, metrics, coverage, reliability, benchmarks
knowledge/        Local research corpus with provenance
reporting/        Markdown report generation
cli/              CLI-first entry points
```

## Evaluation lifecycle

```
Target
  -> Assessment
  -> Test
  -> Experiment
  -> Input / intervention
  -> Model / system response
  -> Context
  -> Observation
  -> Evaluation (PASS / FAIL / ANOMALY / INCONCLUSIVE / NOT_APPLICABLE)
  -> Evidence
  -> Finding (only for FAIL / ANOMALY)
  -> Follow-up experiment
  -> Reproduction
```

Execution status is intentionally separate from evaluation result:

- `COMPLETED`, `PARTIAL_EXECUTION`, `EXECUTION_ERROR` describe execution.
- `PASS`, `FAIL`, `ANOMALY`, `INCONCLUSIVE`, `NOT_APPLICABLE` describe the
  evaluated property.

An execution error never becomes `PASS`.

## Model evaluation

`ExperimentRunner` executes a prompt against a `ModelTarget` and returns
`Evidence`.

- `ModelSafetyRunner` composes `ModelSafetySuite` cases with the
  `SafetyBoundaryEngine`, producing an `Evaluation` and (on
  FAIL/ANOMALY) a `Finding`. Optional semantic refinement can downgrade
  a deterministic `PASS` when an embedding backend disagrees; an
  unavailable backend never changes the result.
- `ModelSecurityRunner` composes `ModelSecuritySuite` cases with the
  deterministic `ModelSecurityEngine`.
- `AdaptiveController` runs bounded, evidence-driven follow-up
  experiments using distinct, scope-preserving prompt variations,
  reusing the baseline and stopping when exhausted.

## System evaluation

An authorized `AIApplicationTarget` returns a `SystemInteraction`.
`AgentAssessmentEngine` runs independent property evaluators over the
same interaction:

authorization, data boundary, action boundary, retrieval integrity,
RAG security, context boundary, tool invocation, tool result boundary,
side effect, identity boundary, output handling, groundedness,
confirmation, and sequence-aware agent trace boundary.

Each evaluator returns its own result and evidence. The aggregate never
replaces the individual results. `NOT_APPLICABLE` checks are excluded
from the aggregate so an inapplicable property cannot degrade an
otherwise passing assessment.

`SystemAdaptiveController` provides a bounded, evidence-driven
follow-up loop for application/RAG/agent targets: it plans
scope-preserving input variations, executes them through the target,
re-evaluates with the same checks, persists experiments and
assessments, and records reproduction status against a source finding.
Execution failures are recorded as `INCONCLUSIVE`, never `PASS`.

## Persistence and lineage

`Database` stores experiments, assessments, findings, and reproductions
in SQLite with additive, idempotent migrations. Experiments carry
`property_name`, `property_expectation`, `comparison_group`,
`evaluator`, assessment columns, and `reproducibility` metadata.

## Research layer

`ExperimentDataset` and `ResearchDataset` provide stable, exportable
views with deterministic SHA-256 fingerprints. Metrics, coverage, and
reliability are descriptive; AIVEXA deliberately does not collapse
results into a single score. Benchmarks are versioned definitions that
can be executed with `BenchmarkRunner` and evaluated with an explicit
evaluator.

## Knowledge layer

`KnowledgeCorpus` is a file-backed corpus with provenance
(source, timestamps, content hash) and deduplication. It is separate
from the experiment database. Follow-up plans can attach related
corpus entry IDs so the research material that informed a decision
remains traceable.

`NvdCveSource` retrieves CVE records from the live NVD CVE API 2.0 with
pagination, incremental last-modified refresh, rate-limit handling, and
stable identifiers. `refresh_corpus` ingests retrieved entries and
produces an explicit `RefreshReport` (added/updated/unchanged, next
start index, errors) so a partial or failed refresh is never presented
as a successful one.

## Execution reliability

`targets/` adapters do not manage retries or timeouts individually.
They delegate to a shared `ExecutionPolicy` (`aivexa/execution.py`) that
provides per-attempt timeouts, an optional overall deadline, bounded
transient-only retries with backoff, typed `ExecutionError`
classification (`CONNECTION`, `TIMEOUT`, `HTTP_STATUS`, `MALFORMED`),
and per-request telemetry. An execution failure is always recorded as an
execution error and never converted into `PASS`.

## Boundaries

- Deterministic and local-first; no LLM-as-a-judge is required.
- Safety (boundary preservation) and security (property violation) are
  distinct but share experiment, evidence, and reporting infrastructure.
- CLI-first; no GUI.
