# Evaluation

_Every number in this document comes from an actual run, and each section
says which code it was measured on. Nothing here is estimated or
illustrative._

## Read this first: which numbers are reliable

A full audit of this project found bugs in the **measuring instruments**, not
only in the runtime. Numbers measured before those fixes are kept below,
because the failure modes they revealed are real, but they are labelled and
should be re-measured before being quoted.

| Section | Measured on | Status |
|---|---|---|
| [Retrieval quality](#results-knowledge-base-previous-revision) (Recall@k, MRR) | previous revision | **Reliable.** Deterministic, no LLM involved, retrieval code unchanged. |
| [Knowledge-base eval](#results-knowledge-base-previous-revision) (single vs multi-agent) | previous revision | **Unreliable - re-run.** Harness never reset conversation memory between questions: each single-agent prompt contained earlier questions and answers; for multi-agent, only the Researcher worker's prompts did. The numerical verifier then accepted about half of fabricated numbers, and unverified last-step answers counted as successes. |
| [Retrieval budget experiment](#experiment-retrieval-budget--search-efficiency-previous-revision) | previous revision | **Unreliable - re-run.** Same memory leak and verifier. |
| [Product analytics budgeting experiment](#experiment-product-analytics-budgeting-previous-revision) | previous revision | **Unreliable - re-run.** No memory leak (that harness did reset), but its governance never ran, and its "numerical correctness" was a tautology. |
| [Verifier measurement](#measurement-numerical-verifier) | this revision | **Reliable.** Deterministic, `scripts/measure_verifier.py`. |
| [Domain smoke runs](#measurement-real-model-smoke-runs-of-the-domain-runtime) | this revision | Reliable as observations; five questions, single runs - not a benchmark. |
| [Test suite](#test-suite) | this revision | Reliable. |

To re-measure: `scripts/run_evaluation.py`, `scripts/run_budget_experiment.py`,
`scripts/run_product_analytics_experiment.py`, `scripts/run_cross_domain_eval.py`.

## Benchmark design

**Knowledge base** (`data/benchmark.json`): ~20 hand-written questions over
the synthetic "Nimbus" corpus, each with `question`, `category`,
`difficulty`, `expected_doc_ids`, `expected_record_ids`, `expected_answer`,
and `expected_tool` where a specific tool is the correct choice:

| Category | Tests |
|---|---|
| `pure_document` | Retrieval + grounded generation from documents alone |
| `structured_record` | Correct tool selection toward `records`, not documents |
| `tool_arithmetic` | Calculator use, correct expression construction |
| `multi_hop` | Combining two sources (a document + a record, or two documents) |
| `out_of_corpus` | Refusing rather than hallucinating when no evidence exists |

**Business domains** (`domains/<name>/benchmark.json`): 8-12 questions each,
adding `expected_numbers` (figures the answer should contain) and, for
questions that ask for a decision, `expected_policy_status`. Every expected
value was checked against the data; this revision corrected five entries that
did not match it (a p-value of 0.015 that is really 0.014, an operations
answer citing a 10% threshold that is really 25%, a reference to tablet data
that does not exist, and two questions with an expected policy status that
never actually asked for a decision).

These are intentionally small - enough to catch category-level patterns and
compare configurations on identical inputs, not enough to report metrics to
two decimal places with any real confidence interval.

## Metrics

**Retrieval** (scored directly against the retriever, independent of any
agent - `evaluation/rag_metrics.py`): Recall@k and MRR, both naive
(bi-encoder only) and reranked.

**Generation**:
- Groundedness - LLM-judge check that claims trace to evidence (`agent/llm_judge.py`)
- Citation correctness - of the citations used, the fraction pointing to an expected document
- Answer verified - whether the final answer passed the loop's deterministic
  citation and numerical checks (`AgentState.answer_verified`)
- Expected-number recall - the fraction of the benchmark's expected figures
  present in the answer, matched at the coarser of the two precisions
  (`agent_metrics.score_expected_numbers`)

A caveat this project's own runs made concrete: **groundedness is necessary
but nowhere near sufficient.** A system that refuses every question scores
~100% grounded. An early multi-agent run did exactly this - the planner
selected no workers, so every answer was a well-grounded "I cannot determine
that." Task success is the metric that catches it, which is why both are
reported.

**Agent**: task success (LLM judge against `expected_answer`), tool selection
accuracy, unnecessary tool calls, step count, latency, policy alignment (the
engine's verdict equals `expected_policy_status`; averaged only over questions
that expect a decision, and a missing verdict counts as a miss), and failure
rate. A single-agent run fails if it never reached a final answer *or* only
reached one that did not pass verification; multi-agent counts a draft its
verifier could not confirm. Those are different events, so compare failure
rates within a mode and use task success across modes.

**Judge failures** are excluded from averages and counted as `unjudged_rows`.
(The product analytics and cross-domain harnesses previously counted a failed
judge call as a task *failure* and as *grounded*.)

## Results: knowledge base (previous revision — WITHDRAWN)

> **Notice:** The earlier LLM-judged quality numbers below were withdrawn after an audit revealed
> that the evaluation harness leaked conversational short-term memory across benchmark questions,
> the numerical verifier accepted fabricated figures (50.9%), and unverified last-step answers were
> counted as successes. These numbers are removed pending a clean re-run.
> To regenerate clean benchmark results: `uv run python scripts/run_evaluation.py`.

20 queries, `qwen2.5:7b-instruct` via Ollama:

| Metric | Single-agent | Multi-agent | Status |
|---|---|---|---|
| Task success rate | [pending rerun] | [pending rerun] | Withdrawn (memory leakage in harness) |
| Groundedness rate | [pending rerun] | [pending rerun] | Withdrawn (memory leakage in harness) |
| Citation correctness | [pending rerun] | [pending rerun] | Withdrawn |
| Tool selection accuracy | [pending rerun] | [pending rerun] | Withdrawn |
| Avg. unnecessary tool calls | [pending rerun] | [pending rerun] | Withdrawn |
| Avg. step count | [pending rerun] | [pending rerun] | Withdrawn |
| Failure rate | [pending rerun] | [pending rerun] | Withdrawn |
| Avg. latency (s) | [pending rerun] | [pending rerun] | Withdrawn |

Search efficiency counts (`evaluation/retrieval_metrics.py`):

| Metric | Single-agent | Multi-agent | Notes |
|---|---|---|---|
| Avg. retrieval calls | 1.45 | 0.50 | Multi-agent searches once per query by structure |
| Avg. unique retrieval queries | 0.85 | 0.50 | |
| Avg. chunks consumed | 4.35 | 1.50 | |
| Retrieval redundancy | 0.30 | 0.00 | Single agent repeated queries across steps |
| Repeated query calls (total) | 12 | 0 | Structural in multi-agent RetrieverWorker |

**Retrieval** (reliable, deterministic): Recall@k 0.98 (naive) -> 0.98 (reranked); MRR 0.92
(naive) -> 0.93 (reranked).

### What still stands, and what is now a hypothesis

**Reranking did essentially nothing here, and that is a real negative
result** (it is unaffected by the bugs). The corpus is ~32 chunks across 31
short documents, with questions phrased closely to the source text, so the
bi-encoder already retrieves the right chunk at rank 1 almost every time.
Cross-encoder reranking earns its cost on large, noisy corpora - which this
corpus is deliberately too small to be.

**Hypothesis to re-test: constraint, not orchestration.** The single agent's
losses coincided with spending its step budget on repeated searches (1.45
retrievals per query, 12 repeated searches), and the budget experiment below
suggested a one-retrieval cap closes most of the gap. The mechanism is
plausible, but the leaked conversation grew with every question in the single
agent's prompts while the multi-agent planner and retriever never saw it, so
the gap may be partly an artefact.

**Tool selection accuracy understates real competence**, because the
metric assumes exactly one canonical tool per question (a trace-verified
observation, unaffected by the bugs):
- q07 ("Which team owns the Ferry service?") - answered correctly from
  retrieval with no tool call, because `arch-ferry` states the owning team in
  prose. Scored a failure.
- q09 ("What is Beacon's uptime SLO?") - used `search_records` rather than
  the expected `lookup_record`, and got the right answer.

**Citation correctness** was computed over different subsets per mode and
flipped direction between runs; it does not support a winner either way.

## Experiment: Retrieval Budget & Search Efficiency (previous revision)

> Measured before this revision's fixes - see "Read this first".

### The question

The multi-agent path's `RetrieverWorker` searches exactly once per query, by
construction, so multi-agent runs under an implicit retrieval budget of 1. If
a single agent capped at one retrieval performs like the multi-agent path,
the orchestration story largely collapses into a constraint story.
`scripts/run_budget_experiment.py` is the falsification attempt.

### What "budget" means in this codebase

| Budget level | Mechanism |
|---|---|
| Per call (candidates) | `top_k_retrieve=8` -> `top_k_rerank=3` |
| Per run, retrievals only | `AgentRuntime(max_retrievals=...)` - the cap this experiment varies |
| Per run, all resources | `ActionBudget` (steps, retrievals, SQL queries, extractions) - used by adaptive budgeting |
| Per run, all actions | `max_agent_steps=8` |
| Context (tokens) | not bounded |

### Results (WITHDRAWN — Pending Clean Rerun)

> **Notice:** The quality metrics (Task success, Groundedness, Citation correctness, Failure rate) below
> were affected by the conversational memory leakage across questions and permissive verification.
> They are removed pending a clean re-run: `uv run python scripts/run_budget_experiment.py`.

Single-agent, 20 queries per condition (`data/budget_experiment_report.json`):

| Metric | budget=1 | budget=2 | budget=3 | unlimited | Status |
|---|---|---|---|---|---|
| Task success | [pending] | [pending] | [pending] | [pending] | Withdrawn (memory leak in harness) |
| Groundedness | [pending] | [pending] | [pending] | [pending] | Withdrawn (memory leak in harness) |
| Citation correctness | [pending] | [pending] | [pending] | [pending] | Withdrawn |
| Failure rate | [pending] | [pending] | [pending] | [pending] | Withdrawn |
| Avg latency (s) | 10.42 | 11.69 | 11.00 | 13.43 | Historical reference |

Search-behaviour counts (mechanical, unaffected by judge metrics):

| Metric | budget=1 | budget=2 | budget=3 | unlimited |
|---|---|---|---|---|
| Avg retrieval calls | 0.65 | 1.05 | 1.05 | 1.45 |
| Avg chunks consumed | 1.95 | 3.15 | 3.15 | 4.35 |
| Retrieval redundancy | 0.00 | 0.14 | 0.38 | 0.32 |
| Repeated query calls (total) | 0 | 3 | 8 | 10 |

Out-of-corpus questions answered correctly: 3/3 at budget=1, 1/3 unlimited.

### Reading it now

The search-behaviour columns (retrieval calls, chunks, redundancy, repeats)
are mechanical counts of what the agent did and are the most robust part of
this table - though the leaked conversation changed the prompts that drove
that behaviour. The quality columns (task success, citation correctness,
failure rate) are exactly what the memory leak and permissive verifier could
distort, so the headline claims - "more retrieval freedom made every
dimension worse" and "a retrieval cap acts as a hallucination control"
(the out-of-corpus 3/3 vs 1/3) - are hypotheses with a visible mechanism,
pending a clean re-run. Even a clean result would not show that budget=1 is
generally right: most of these questions need a single retrieval, and a
corpus requiring multi-document synthesis should punish a cap of 1.

## Experiment: Product Analytics Budgeting (previous revision)

> Measured before this revision's fixes - see "Read this first". This harness
> did reset conversation memory per question, but the problems below make the
> quality columns uninformative.

> **Notice:** The quality metrics below were uninformative or tautological (see inspection notes).
> They are removed pending a clean re-run: `uv run python scripts/run_product_analytics_experiment.py`.

12 questions from `domains/product_analytics/benchmark.json`, `qwen2.5:7b-instruct`,
single run per regime (`data/product_analytics_experiment_report.json`):

| Metric | Unconstrained | Fixed | Adaptive | Status |
|---|---|---|---|---|
| Task success rate | [pending] | [pending] | [pending] | Withdrawn (judge failures counted as task failure) |
| "Numerical correctness" | [pending] | [pending] | [pending] | Withdrawn (tautological check on already-verified output) |
| Groundedness | [pending] | [pending] | [pending] | Withdrawn (judge failures counted as grounded) |
| "Abstention precision" | [pending] | [pending] | [pending] | Withdrawn (evaluated on single question pa-11) |
| Avg steps per task | 5.00 | 3.58 | 3.33 | Mechanical step count |
| Total SQL queries | 29 | 17 | 16 | Mechanical query count |
| Unnecessary tool calls (total) | 8 | 2 | 4 | Exploration count |
| Mean latency (s) | 18.27 | 12.15 | 10.39 | |
| Median latency (s) | 11.21 | 9.42 | 9.61 | |
| Latency std. dev. (s) | 17.13 | 8.51 | 6.95 | |

What the report file itself shows on inspection:

- **Governance never ran.** The one decision question (pa-10) has no policy
  evaluation in any regime - the model never attached a decision, and the
  harness excluded missing evaluations from policy alignment instead of
  counting them as misses. The rollout rules it would have used also pooled
  p-values across experiments and ignored the direction of the effect.
- **"Numerical correctness" was a tautology.** It re-ran the verifier that the
  loop already applied before accepting an answer, and scored 1.0 on all 36 rows -
  including a run that ended without a final answer. It is now replaced by
  expected-number recall.
- **"Abstention precision" was one question.** It was recall on pa-11 alone,
  so 1.00 vs 0.00 is that single question failing under the adaptive budget.
- A failed judge call counted as a task failure and as grounded.

What survives is the resource side: under fixed and adaptive budgets the
agent took fewer steps, ran fewer SQL queries, and had lower and less spread
latency than unconstrained. Those are counts from one run per regime, not an
effect size, and the classifier that picked adaptive budgets then matched
keywords as substrings ("count" inside "account"), which has since been fixed.

## Measurement: numerical verifier

`uv run python scripts/measure_verifier.py` - deterministic, no LLM. The
ledger holds 12 daily checkout rows and one A/B test result (rates, a signed
difference of +1.333 pp, a p-value). The baseline is the verifier at commit
`b13a25d`, loaded from git history.

<!-- BEGIN_VERIFIER_TABLE -->
| | Baseline (b13a25d) | Current |
|---|---|---|
| Random fabricated claims accepted (2,000; percentages 0.5-60%, gaps 0.2-15 pp, "fell" 1-90%) | 1,018 (50.9%) | 59 (2.9%) |
| Out-of-range claims accepted (2,000; 45-95%, 5-40 pp, counts 3,000-99,000 - no real value or absolute difference in range) | 67 (3.4%) | 0 (0.0%) |
| Plausible-range percentages (2,000; 1.4-4.3% range) | 2,000 (100.0%) | 136 (6.8%) |
| Percentage-point gaps (2,000; 0.1-3.0 pp range) | 2,000 (100.0%) | 479 (23.9%) |
| Relative changes (2,000; whole-number falls 1-60%) | 2,000 (100.0%) | 1,009 (50.4%) |
| Entity swaps accepted (6 trials, phrased multiple ways) | 6 (100.0%) | 0 (0.0%) |
| Synonym swaps accepted (6 trials, phone/workstation/laptop) | 6 (100.0%) | 0 (0.0%) |
| Metric swaps accepted (6 trials, checkout vs purchase / DTI vs score) | 5 (83.3%) | 0 (0.0%) |
| Development probes: adversarial Set 1 (used while building fix, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Development probes: adversarial Set 2 (used while building fix, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Development correct paraphrases (16 trials, false rejection control) | 16/16 (100.0%) | 16/16 (100.0%) |
| Held-out adversarial probes (Set 3, pre-fix, 15 trials) | 15/15 (100.0%) | 2/15 (13.3%) |
| Held-out correct claims (Set 3, pre-fix, 15 trials, false rejection control) | 15/15 (100.0%) | 13/15 (86.7%) |
| "Variant B fell by 1.33 percentage points" (evidence: +1.333) | accepted | rejected |
| Version number accepted as valid ("...iOS 17 update") | accepted | accepted |
| Benchmark correct answers (3 trials, false rejection control) | 3/3 (100.0%) | 3/3 (100.0%) |
<!-- END_VERIFIER_TABLE -->

The random set's residual 4.0% is mostly values that coincide with real ones
at the precision written (e.g. a random "2.8%"), which is correct behaviour.
The baseline's acceptances came from a flat ±1.5 pp tolerance and from
searching every pair of numbers for a matching difference. What neither
version checks is meaning - see the limitations in `agent/verification.py`.

## Measurement: citation support verification

`uv run python scripts/measure_citations.py` - deterministic, no LLM. Evaluates
whether citation verification correctly rejects unsupported claims and citations
pointing to irrelevant or fabricated context across 60 hand-labelled pairs.

<!-- BEGIN_CITATION_TABLE -->
| Metric | Baseline (existence check) | Lexical Fallback | NLI Entailment Verifier |
|---|---|---|---|
| Benchmark pairs | 60 | 60 | 60 (30 supported, 30 unsupported) |
| Recall (supported citations preserved) | 100.0% (30/30) | 100.0% (30/30) | 93.3% (28/30) |
| Out-of-domain hallucinations rejected | 0.0% (0/15) | 100.0% (15/15) | 100.0% (15/15) |
| Contradictory citations accepted (false pos.) | 100.0% (15/15) | 20.0% (6/30) | 10.0% (3/30) |
| Precision | 50.0% | 83.3% | 90.3% |
| Overall Accuracy | 50.0% | 90.0% | 91.7% |
| Held-out pairs (20 pairs: 10 supported, 10 unsupported) | - | Precision 83.3%, Recall 100.0% | Precision 100.0%, Recall 100.0% |
<!-- END_CITATION_TABLE -->

## Measurement: real-model smoke runs of the domain runtime

Five decision questions - cr-02 (expected ALLOWED), cr-05 (BLOCKED), ops-04
(BLOCKED), mm-03 (BLOCKED), pa-10 (REQUIRES_APPROVAL) - through the full
stack: `qwen2.5:7b-instruct`, the domain's MCP server as a subprocess, its
policy engine, adaptive budgets, no LLM judge. One run per configuration, so
treat single-question differences as noise; the value of these runs is the
failure modes they exposed, each of which led to a fix.

| Run | Code | Verified answers | Decisions produced | Verdict = expected |
|---|---|---|---|---|
| 1 | start of this revision's runtime work | 2/5 | 1/5 | 1/5 (pa-10, for the wrong reasons) |
| 2 | + no citation rule for tool-only agents, calculator results ground percentages, per-domain decision guide | 3/5 | 0/5 | 0/5 |
| 3 | + policy thresholds in the prompt, numbers in field names grounded, decision extraction fallback | 5/5 | 4/5 | 4/5 |
| 4 | same as 3; mm-03 question reworded to actually ask for a decision | 4/5 | 3/5 (2 extracted, 1 attached) | 3/5 |
| 5 | + calculator evaluates comparisons | 5/5 | 4/5 (3 extracted, 1 attached) | 4/5 |

What each finding was, and what it changed:

1. **A tool-only agent was told to cite documents.** The prompt's "cite
   retrieved documents using [n] tags" rule was shown even with no retrieval;
   the model appended invented `[3] [4]` tags to SQL-based answers, which then
   failed citation verification. The rule is now shown only when retrieval
   exists.
2. **A correctly computed percentage was rejected.** The prompt tells the
   model to compute derived numbers with the calculator; it computed
   `3000 / 5000 * 100 = 60` and wrote "60%", which the unit-aware verifier
   rejected because a bare 60 is not a percentage. Calculator results may now
   ground percentages.
3. **Cross-domain examples were copied.** The generic prompt listed example
   actions from all four domains; on a product-analytics rollout the model
   proposed `rolling_restart` with `service_name="EXP-204"`. Examples now come
   from each domain's manifest (`decision_guide`), so the core prompt holds no
   domain vocabulary.
4. **Invented criteria burned the step budget.** Not knowing the real
   limits, the model wrote unsupported figures into its credit answers (a
   "0.50" limit, a "30.49%" post-increase ratio), failed verification
   repeatedly and hit the step limit. The engine's thresholds are
   now stated in the prompt (and are always grounded). A related false
   rejection - "no delinquencies in the last 90 days", when the evidence field
   is `delinquencies_90d` - was fixed by grounding numbers in field names.
5. **Decisions were described but not attached.** Across runs 1-2 the model
   attached a `proposed_decision` in 1 of 10 runs (1 of the 7 that reached a
   final answer), once writing "the proposed decision is to rollout_variant_b
   with an experiment_id of 'EXP-204'" in prose while leaving the field null.
   A follow-up extraction call with an all-required schema now recovers the
   decision; the engine still judges it, and the trace records
   `proposed_decision_source`. In runs 4-5, five of the seven decisions came
   from extraction.
6. **The calculator could not compare.** Asked to check thresholds, the model
   issued calls such as `(670 >= 650) and (0.48 <= 0.43)`, which failed; in run
   4, seven calculator calls on pa-10 (six of them failing comparisons) ended
   the run at the step limit. The calculator now evaluates comparisons and
   `and`/`or`/`not`, never mixing booleans into arithmetic. In run 5 every
   such call succeeded and no question needed more than 4 steps.
7. **Wrong prose, correct verdict.** Several answers were wrong in ways
   verification cannot see: a 100% blast radius described as "within the
   threshold"; an invoice with a $500 arithmetic discrepancy that "can be
   approved for payment"; and a recommendation to proceed with a change to
   a service during its active P0 incident. Each time the policy engine,
   reading evidence, returned the correct restrictive verdict. That is the
   design working as intended - and the reason decisions never read the
   answer text.
8. **A miscounted small number went unchecked.** One answer stated "1
   delinquency in the last 90 days" for an account whose evidence says 0.
   Integers from 0 to 10 are not treated as claims (they are mostly ordinals
   in prose), so verification passed it; the delinquency rule read the real 0.

## Observed failure modes (earlier revisions)

From running the project against Ollama, not hypothetical. Each names the
fix it motivated.

1. **Citation-avoidance loop.** On one run, the model answered a fully
   retrieved factual question *without* citation tags three times in a row,
   rejected each time, before citing on attempt 4 of an 8-step budget. This
   motivated the concrete example in the loop's corrective message.
2. **Verification bypass via an overlapping tool path.** The single agent had
   two ways to search documents and only one fed `citation_map`; a fabricated
   revenue figure with zero citations passed verification via the other. See
   ARCHITECTURE.md's "Why `retrieve` is MCP-mediated".
3. **Redundant tool calls on a paraphrased query.** Five reworded searches,
   three returning the identical top result, until the step limit ended the
   run with no answer - inefficiency severe enough to become a correctness
   failure.
4. **Structured-output violations that aborted whole runs.** A `null` search
   query, `confidence: 70` for a 0-1 field, and the same class of failure in
   the verifier and memory-write steps. Each now degrades instead of
   crashing: a missing query falls back to the question, percentage-style
   confidence is rescaled, a failed verifier marks the answer unverified, a
   failed memory write never discards an answer.
5. **A vague worker description silently disabled the multi-agent path.**
   "Specialists" was read as human job titles, so no worker was ever
   selected. Naming the two concrete workers and a default fixed it: a tool or
   worker description is a prompt.
6. **The benchmark itself was wrong.** `expected_tool` named
   `records_lookup` while the server registers `lookup_record`, so tool
   selection accuracy could never pass - caught because 5 of 6 "failures" had
   `task_success: true`. **A metric disagreeing with the traces is a reason to
   audit the metric.**

## Failures found by auditing (this revision)

Found by reading the code and then confirming with tests, not by a model run.
Each has a regression test.

- **Measurement**: no memory reset between benchmark questions (single-agent
  and the multi-agent Researcher worker); judge failures scored as
  pass/fail; a tautological "numerical correctness"; policy alignment averaged
  over questions that expect no decision; unverified last-step answers counted
  as successes.
- **Governance**: domain rules read facts from the model's own proposal (a
  self-granted `override_authorized: true` lifted a production freeze); took
  the first matching number in the ledger regardless of which account or
  invoice it belonged to; treated missing facts as "no violation"; matched a
  short list of exact action strings, so other spellings skipped every rule;
  read flags with `bool()`, so the string "false" was true; accepted a
  1-cent invoice discrepancy; fired the SLA rule on the weakest week instead
  of the monthly average; and let manifests drift from code.
- **Runtime**: one budget controller shared by every run; evidence payloads
  that could be mutated after recording; a numerical-verification exemption
  for any answer containing a phrase like "not available"; duplicate tool
  names silently overwriting each other; SQL validation that rejected
  harmless literals, allowed unbounded results and leaked connections on
  errors; vision tools that read arbitrary absolute paths and invented
  `settlement_status="unpaid"` for documents without one; a remediation
  simulator that assumed every service has 12 nodes; long-term memory that
  injected memories into prompts however unrelated they were; and an HTTP API that shared
  one caller's conversation with the next.

## What this does and doesn't demonstrate

Supported: the deterministic controls behave as specified (test suite); the
current numerical verifier rejects far more fabrications than the original
(measured); retrieval quality and the lack of a reranking benefit on this
corpus (deterministic); and the failure modes above, which are real and
reproducible.

Not supported until re-measured: any single- vs multi-agent quality
comparison, any claim that retrieval caps improve answer quality, and any
budget-regime quality comparison on the product analytics benchmark. Not
supported at all by benchmarks this small: general claims about agent
architectures, model sizes or RAG techniques, or any difference under
~0.10-0.15.

## Test suite

`uv run pytest` runs the suite with a scripted fake LLM driving the real
agent loop, real MCP server functions (in-process, and over the MCP protocol
for discovery), real DuckDB databases and the real domain policies. It
covers: evidence immutability; fact resolution and the restrict/permit
asymmetry; policy precedence and fail-closed behaviour; numerical and
citation verification (including the direction, unit, refusal-phrase and
field-name cases); per-run budgets; SQL validation and truncation; path
confinement in the vision tools; the domain loader's manifest checks; each
domain's decision flows plus regression tests for every governance bypass
listed above; decision extraction; memory; and the evaluation metrics.
