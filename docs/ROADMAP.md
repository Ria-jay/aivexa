# AIVEXA — Roadmap and Completion Checklist

This file tracks the agreed AIVEXA scope against the actual implementation.
It is maintained from real behaviour and tests, not from file existence.

Status legend:

- `[x]` Complete and verified
- `[~]` Implemented but insufficiently validated
- `[ ]` Missing
- `[!]` Blocked (specific reason)

Last reconciled against commit: see `git log -1`.

---

## A. Core experiment and evidence architecture

- [x] Target → Assessment → Experiment → Evaluation → Evidence → Finding lifecycle.
- [x] Evidence provenance tied to experiment and finding.
- [x] Findings reproducible via follow-up experiments (`reproduction.py`).
- [x] Errors, timeouts, partial executions, and inconclusive outcomes explicit.
- [x] Execution failures never become PASS (`SafetyExecutionStatus`, runner guards).
- [x] Results, persisted records, metrics, and reports agree (sqlite + dataset layers).
- [x] Safe schema migrations (`Database._initialize` additive migrations).
- [x] Interrupted/partial workflows handled (PARTIAL_EXECUTION, errors tuple).

## B. Model safety evaluation

- [x] Refusal / safe-redirection classification.
- [x] Behaviour taxonomy: REFUSAL, SAFE_REDIRECTION, GENERAL_EDUCATION,
      PROCEDURAL, ACTIONABLE, COMPLIANCE, AMBIGUOUS.
- [x] Deterministic topic relevance (case-supplied `topic_keywords`).
- [x] Detection of subject-changing / unrelated answers.
- [x] Paraphrase, framing, multi-turn, and context robustness cases.
- [x] Unusual safe-refusal wording (broad refusal patterns).
- [x] Confidence calibration and explicit uncertainty in evidence.
- [x] Regression tests for unrelated-answer PASS and safe-refusal ANOMALY.
- [x] Consistent FAIL / ANOMALY / INCONCLUSIVE / execution-error distinction.
- [x] Optional model-backed semantic evaluator with honest unavailability
      (`semantic_relevance.py`, never converts evaluator error into PASS).
- [x] Relevant refusals count as boundary preservation; the topic
      vocabulary includes request/safety context so valid refusals are
      not misclassified as off-topic, while genuinely unrelated answers
      still are.
- [x] Refusal disclaimers followed by actionable content are not treated
      as safe refusals (refusal-content contradiction / actionable
      escalation).

## C. Model security evaluation

- [x] Prompt-injection resistance.
- [x] Instruction / policy boundary.
- [x] Sensitive-information disclosure.
- [x] Protected-context / instruction extraction attempts.
- [x] Security-sensitive output handling.
- [x] Robustness under paraphrase/framing/multi-turn pressure.
- [x] Conflicting / untrusted instructions (adversarial case library).
- [x] Distinct result vs execution-error handling.
- [x] CLI-reachable (`aivexa model-security run`).

## D. Adaptive probing and investigation

- [x] Follow-up hypotheses derived from observed evidence.
- [x] Evidence-driven next-experiment selection.
- [x] Distinct probe generation and prompt de-duplication.
- [x] Baseline reuse across cycles.
- [x] Experiment history / lineage persistence.
- [x] Reproduction attempts for anomalies and failures.
- [x] Stopping conditions, exhaustion handling, and cycle budgets.
- [x] Explicit reasons for each action and stopping decision.
- [x] Scope-preserving generated probes (strategies do not broaden scope).
- [x] Adaptive vs fixed-run efficiency/coverage measurement
      (`research/adaptive_efficiency.py`).

## E. Target discovery, profiling, and providers

- [x] Registry CRUD and configuration validation.
- [x] Capability/interface profiling before assessments
      (`targets/profiling.py`).
- [x] Provider abstraction independent of Ollama (`targets/factory.py`).
- [x] OpenAI-compatible HTTP model adapter (`targets/openai_compatible.py`).
- [x] Target metadata, capabilities, configuration, and limitations.
- [x] Explicit rejection of unsupported providers/endpoints.
- [x] Credential handling via environment variables only.
- [x] Authorization/scope checks before execution.
- [ ] Additional built-in providers beyond Ollama/OpenAI-compatible/Gemini
      (not implemented; see Limitations).

## F. AI application, RAG, and agent security

- [x] Untrusted retrieved content / indirect prompt injection.
- [x] Retrieval manipulation / poisoned / conflicting context.
- [x] Sensitive data disclosure and cross-context leakage.
- [x] Access-control boundaries for retrieved documents.
- [x] Trusted-vs-untrusted instruction separation.
- [x] Response supported by retrieved evidence (`groundedness.py`).
- [x] Tool-call authorization boundaries.
- [x] Unauthorized / unintended action attempts.
- [x] Untrusted instructions through tools / external content.
- [x] Excessive permissions / sensitive-data exposure.
- [x] Consequential-action confirmation boundary (`confirmation.py`).
- [x] Cross-component boundaries (model/tools/data/application).
- [x] Sequence-aware agent trace policy evaluation, including tool
      allow-lists, argument scope, untrusted instruction sources,
      confirmation ordering, and untrusted tool results
      (`evaluation/trace_boundary.py`, `evidence/agent.py`).
- [x] Controlled mock/local test environments, including a scripted
      mock agent target (`targets/mock_agent.py`).

## G. Adaptive evaluation across model and system targets

- [x] Adaptive investigation integrated with model evaluations.
- [x] Target capabilities → case selection.
- [x] Hypotheses/observations → experiments → evidence → findings.
- [x] Adaptive investigation for system/application/RAG/agent targets
      (`evaluation/system_adaptive.py`): plans, executes, evaluates,
      persists, and reproduces follow-ups with scope enforcement,
      budgets, de-duplication, and stopping conditions.
- [x] End-to-end test proving a system observation produces an executed
      follow-up experiment, evaluation, and persisted outcome.

## H. Research corpus and knowledge integration

- [x] Local knowledge corpus with provenance, dedup, timestamps
      (`knowledge/corpus.py`).
- [x] Search/retrieval that informs hypotheses and follow-up experiments.
- [x] Traceable links between corpus material and decisions.
- [x] Resilient parsing and explicit failure reporting.
- [x] Live NVD CVE API 2.0 refresh adapter (`knowledge/nvd.py`) with
      pagination, incremental last-modified refresh, optional API key,
      timeouts, bounded retries, rate-limit handling, and resumable
      start indices; refresh is orchestrated by `knowledge/refresh.py`.
- [x] Live NVD retrieval verified in this environment (see README).
- [x] No duplicate database — corpus is file-backed and separate from the
      experiment database.

## I. Research, benchmarking, and evaluator reliability

- [x] Repeatable benchmark runs with recorded configurations
      (`research/benchmark_runner.py`).
- [x] Cross-target comparison (descriptive, no ranking).
- [x] Fixed-suite vs adaptive comparison (`research/adaptive_efficiency.py`).
- [x] Coverage and outcome distributions.
- [x] FP/FN/inconclusive/execution-failure tracking on labelled sets
      (`research/accuracy.py`).
- [x] Evaluator reliability / agreement.
- [x] Reproducibility fingerprints and provenance.
- [x] Documented benchmark limitations and bias.
- [x] Exportable datasets and metrics.

## J. Reporting and findings

- [x] Generic experiment report exposed in CLI (`report experiment`).
- [x] Finding report with reproduction status.
- [x] Model safety assessment report.
- [x] Coverage and limitations sections.
- [x] Modular generation, not hardcoded to one provider.

## K. CLI, configuration, and usability

- [x] Consistent command groups: target, adversarial, model-safety,
      model-security, assess, report, research.
- [x] Helpful usage and errors; predictable exit codes.
- [x] `--json` machine-readable outputs.
- [x] Configuration validation and sensible defaults.
- [x] No hidden prerequisites documented capabilities.

## L. Security, scope, privacy, and reliability

- [x] Authorization/scope enforcement before execution.
- [x] No hardcoded credentials; env-based secret resolution.
- [x] Network timeouts on every outbound request.
- [x] No silent error → PASS conversion.
- [x] Shared provider-aware execution policy with per-attempt timeouts,
      bounded transient-only retries, backoff, typed error
      classification, and telemetry (`aivexa/execution.py`).
- [x] No unintended data collection/transmission.

## M. Packaging, documentation, and maintainability

- [x] Installable package with declared optional dev dependencies.
- [x] Quick start and CLI reference in README.
- [x] Architecture documentation (`docs/ARCHITECTURE.md`).
- [x] Provider matrix and limitations (`docs/PROVIDERS.md`).
- [x] Roadmap/checklist (this file).
- [x] No stale backups/placeholders.

## N. Integration and release readiness

- [x] Full suite run with `python -m pytest -q`.
- [x] End-to-end workflows across model safety, security, application,
      RAG, agent, adaptive, research, and reporting.
- [x] Regression coverage for known evaluator bugs.

---

## Known limitations (not disguised as complete)

1. No LLM-as-a-judge used as ground truth. The optional semantic evaluator
   is embedding-based and advisory; when its endpoint is unavailable the
   evaluation reports unavailability and never returns PASS from an error.
2. General relevance without a declared topic cannot be assessed
   deterministically; relevance checking is enabled only when a case
   supplies `topic_keywords`.
3. Live NVD CVE refresh is implemented and was verified against the live
   NVD API 2.0 in this environment. Other advisory sources (CISA KEV, GHSA,
   vendor feeds) are not bundled; `HttpCorpusSource` plus the source
   protocol allow them to be added. A refresh is only reported successful
   when entries were actually retrieved.
4. Adaptive investigation loops are integrated for both model targets and
   system/application/RAG/agent targets.
5. Provider support is limited to targets that can actually be exercised:
   Ollama, OpenAI-compatible HTTP, and the Gemini REST application adapter.
6. Tool-use evaluation works on observable tool calls/traces supplied by
   the target; AIVEXA does not execute arbitrary agent tools itself.
