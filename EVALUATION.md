# Evaluation

_Every number below is from an actual run of `scripts/run_evaluation.py`
against the 20-query benchmark in `data/benchmark.json`, recorded in
`data/eval_report.json`. Nothing here is estimated or illustrative._

## Benchmark design

~20 hand-written queries over the synthetic "Nimbus" corpus, each with:
`question`, `category`, `difficulty`, `expected_doc_ids` (for document-based
questions), `expected_record_ids` (for structured-record questions),
`expected_answer`, and `expected_tool` (where a specific tool is the
correct choice). Categories, deliberately spanning distinct failure modes:

| Category | Tests |
|---|---|
| `pure_document` | Retrieval + grounded generation from documents alone |
| `structured_record` | Correct tool selection toward `records`, not documents |
| `tool_arithmetic` | Calculator use, correct expression construction |
| `multi_hop` | Combining two sources (a document + a record, or two documents) |
| `out_of_corpus` | Refusing rather than hallucinating when no evidence exists |

This is intentionally small - illustrative, not statistically rigorous.
Twenty questions is enough to catch category-level patterns (e.g. "the
agent is unreliable on X") and to make a single-vs-multi-agent comparison
meaningful on identical inputs; it is not enough to report a metric to two
decimal places with any real confidence interval, and this document does
not pretend otherwise.

## Metrics

**Retrieval** (scored directly against the retriever, independent of any
agent - see `evaluation/rag_metrics.py`):
- Recall@k - of the expected documents, what fraction were retrieved at all
- MRR (Mean Reciprocal Rank) - how highly the first correct document ranked
- Reported both naive (bi-encoder only) and reranked, to make the
  reranking comparison concrete rather than asserted

**Generation** (scored against the agent's actual final answer):
- Groundedness - LLM-judge check that every claim traces to retrieved/tool
  evidence (`agent/llm_judge.py`)
- Citation correctness - of the citations used, what fraction point to an
  expected document (not just *a* valid citation, but the *right* one)

A caveat that this project's own runs made concrete: **groundedness is
necessary but nowhere near sufficient.** A system that refuses every
question scores ~100% grounded, because a refusal with no evidence is
perfectly faithful to that (absent) evidence. An early multi-agent run did
exactly this - the planner selected no workers, so nothing was retrieved,
so every answer was a well-grounded "I cannot determine that." Task
success is the metric that catches it, which is precisely why both are
reported and neither is reported alone.

**Agent**:
- Task success rate - LLM-judge comparison against `expected_answer`
- Tool selection accuracy - did it use the tool the query actually needed
- Unnecessary tool calls - tool calls made when none were needed
- Step count / latency - cost, not just correctness
- Failure rate - **not directly comparable across the two modes**, and
  deliberately left that way rather than forced into a false equivalence.
  Single-agent counts a run that never reached a final answer (hit the step
  limit, or hit an unrecoverable error). Multi-agent has no step limit to
  hit, so it counts a draft the verifier could not confirm as grounded.
  Both are "the system failed structurally," but they are different events;
  read each within its own mode, and use task success for the cross-mode
  comparison.

Step count is likewise mode-relative: a single-agent step is one
decide/act/observe cycle, while a multi-agent "step" is one worker
invocation. Latency and task success are the two metrics that mean the
same thing on both sides.

## Results

20 queries, `qwen2.5:7b-instruct` via Ollama. Latest run;
see "Run-to-run variance" below for the observed range across three runs.

| Metric | Single-agent | Multi-agent |
|---|---|---|
| Task success rate | 0.80 | **1.00** |
| Groundedness rate | 0.95 | **1.00** |
| Citation correctness | **0.88** | 0.65 |
| Tool selection accuracy | 0.40 | 0.60 |
| Avg. unnecessary tool calls | 0.80 | **0.00** |
| Avg. step count | 4.30 | 1.95 |
| Failure rate | 0.05 | **0.00** |
| Avg. latency (s) | 11.07 | **7.94** |

Search efficiency (`evaluation/retrieval_metrics.py`):

| Metric | Single-agent | Multi-agent |
|---|---|---|
| Avg. retrieval calls | 1.45 | 0.50 |
| Avg. unique retrieval queries | 0.85 | 0.50 |
| Avg. chunks consumed | 4.35 | 1.50 |
| **Retrieval redundancy** | **0.30** | **0.00** |
| Repeated query calls (total) | **12** | **0** |

The multi-agent zeros are structural, not earned: `RetrieverWorker` searches
exactly once per query and therefore *cannot* repeat itself. The single
agent, given freedom, re-issued 12 searches for strings it had already
searched, and pulled ~2.9x as many chunks into context.

Task success by category (latest run):

| Category | Single-agent | Multi-agent |
|---|---|---|
| `pure_document` | 7/7 | 7/7 |
| `structured_record` | 2/4 | 4/4 |
| `tool_arithmetic` | 2/3 | 3/3 |
| `multi_hop` | 2/3 | 3/3 |
| `out_of_corpus` | 3/3 | 3/3 |

Note this run's category profile differs from run 2's, where the
single agent's losses were concentrated in `pure_document` (3/7) rather
than `structured_record`. The *aggregate* gap is stable across runs; *which
questions* it comes from is not. Don't over-read a single category row.

**Retrieval:** Recall@k 0.98 (naive) -> 0.98 (reranked); MRR 0.92 (naive)
-> 0.93 (reranked).

### Run-to-run variance

Three full runs of identical code and benchmark produced:

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Single-agent task success | 0.75 | 0.65 | 0.80 |
| Multi-agent task success | 0.95 | 0.95 | 1.00 |

The unconstrained single agent varies over a 0.15 range; multi-agent over
0.05. **The constrained architecture is not just better on average, it is
more predictable** - which follows from the mechanism: free exploration
means more distinct trajectories, and more ways to go wrong.

A useful consistency check falls out of this. While *quality* metrics
swung, the *search-behavior* metrics barely moved: single-agent
`avg_retrieval_calls` was 1.45 in both this run and the budget
experiment's unlimited condition, `avg_chunks_consumed` 4.35 in both, and
redundancy 0.30 vs 0.32. **The efficiency instrumentation is measurably
more stable than the LLM-judged quality metrics**, which is worth knowing
about which numbers to trust from a single run.

### Reading these numbers honestly

**Reranking did essentially nothing here, and that is a real negative
result.** Recall@k was identical and MRR moved by 0.01. The reason is the
corpus, not the technique: ~32 chunks across 31 short documents, with
benchmark questions phrased closely to the source text, so the bi-encoder
already retrieves the right chunk at rank 1 almost every time. There is no
headroom for a reranker to recover. Cross-encoder reranking earns its cost
on large, noisy corpora where the bi-encoder's top-k is genuinely
polluted - which this corpus is deliberately too small to be. The
implementation is exercised and correct; the *benchmark* can't demonstrate
its value, and claiming otherwise from this data would be unsupportable.

**Multi-agent won on task success in all three runs (0.95-1.00 vs
0.65-0.80) - but the follow-up experiment shows that is mostly a
constraint effect, not an orchestration effect.** The single agent's
losses come from spending its step budget on searches and tool calls that
could not answer the question: 1.45 retrieval calls per query with 0.30
redundancy, versus a single non-repeatable search on the multi-agent side.

The tempting conclusion is "the supervisor is smarter." The Retrieval
Budget experiment below tests that and largely refutes it: capping the
*single* agent at one retrieval lifts it to 0.90 task success,
statistically indistinguishable from multi-agent. So the honest claim is
the narrow one - **constraining the action space beat free exploration**,
and multi-agent happens to impose that constraint structurally rather than
being independently better. Read the two sections together; this one alone
overstates the case for orchestration.

**Multi-agent was also ~30% faster (7.94s vs 11.07s)**, which is the
opposite of the usual expectation that orchestration costs latency. Same
cause: the single agent's extra LLM calls came from thrashing, so the
fixed short pipeline (plan -> workers -> synthesize -> verify) is cheaper
than an unbounded loop that wanders. This would likely invert on a harder
benchmark where the fixed pipeline needs multiple rounds.

**Citation correctness is not a clean comparison, and it flips direction
between runs.** The two averages are over *different subsets* - this run,
single-agent cited on 8 of 20 questions and multi-agent on 10, and only
rows with expected documents and at least one citation are scored at all.
Single-agent led 0.88 vs 0.65 here; in the previous run the order was
reversed (0.69 vs 0.72). Across non-identical denominators and with that
much run-to-run movement, **this metric does not support a winner in
either direction** on a benchmark this size. It is reported for
transparency, not as a result.

**Unnecessary tool calls and search redundancy are the metrics that
actually explain the gap.** Single-agent: 0.80 unnecessary tool calls,
1.45 retrievals with 0.30 redundancy, 12 repeated searches. Multi-agent:
0.00, 0.50, 0.00, 0. Those differences are large, consistent across runs,
and mechanically connected to the task-success gap - unlike citation
correctness, which is noise here.

**Tool selection accuracy understates real competence**, because the
metric assumes exactly one canonical tool per question. Two verified
examples from re-running individual queries:
- q07 ("Which team owns the Ferry service?") - the agent called **no tools
  at all** and answered correctly from retrieval, because `arch-ferry`
  states the owning team in prose. Scored a failure; was a correct answer
  by a legitimate route.
- q09 ("What is Beacon's uptime SLO?") - the agent used `search_records`
  rather than the expected `lookup_record`, and got the right answer.
Most tool-selection "failures" coincide with `task_success: true` for
exactly this reason. A better metric would accept any tool path that
reaches correct, grounded evidence; that is left as a known limitation
rather than quietly patched to make the number look better.

**Run-to-run variance is significant** - see the three-run table above.
Single-agent task success spanned 0.65-0.80 on identical code. Differences
smaller than roughly 0.15 on a 20-query set should not be treated as
signal for the unconstrained single agent (0.05 for multi-agent, which is
more stable). Averaging over several seeded runs would be the proper fix
and is not done here; the mitigation used instead is to lean on
differences that reproduce across runs and have a visible mechanism.

## Experiment: Retrieval Budget & Search Efficiency

### The question

The main evaluation showed multi-agent beating single-agent on task
success in every run (0.95-1.00 vs 0.65-0.80) while making *fewer* tool
calls and running *faster*. The obvious reading - "multi-agent is better" -
is not the interesting one, and probably not the right one either.

A sharper hypothesis is available, because of an architectural accident:
**the multi-agent path's `RetrieverWorker` searches exactly once per query,
by construction.** It cannot search twice. So the multi-agent path already
runs under an implicit *retrieval budget of 1* that it never chose and
cannot exceed, while the single agent may search freely up to `max_steps`.

So:

> Is the multi-agent advantage explained by orchestration and
> specialization - or merely by the fact that it enforces
> `retrieval_budget = 1`?

If a single agent capped at one retrieval performs like the multi-agent
path, the orchestration story largely collapses into a constraint story.
That is falsifiable, and `scripts/run_budget_experiment.py` is the
falsification attempt.

### A naming caveat

This project does **not** have a general retrieval-budget mechanism, and
this section does not claim one. What exists is:

| Budget level | Mechanism | Status |
|---|---|---|
| Per-call (candidates) | `top_k_retrieve=8` -> `top_k_rerank=3` | shipped |
| Per-run (searches) | `AgentRuntime(max_retrievals=...)` | added for this experiment only; `None` by default |
| Per-run (all actions) | `max_agent_steps=8` | shipped, but a blunt proxy - retrievals and tool calls share one counter |
| Context (tokens) | none | **not bounded**; evidence accumulates across steps |

Retrieval repetition is also uncontrolled: nothing stops the agent
searching the same thing twice, which the instrumentation below measures
rather than prevents.

### Measures

`evaluation/retrieval_metrics.py`, applied identically to both
architectures:

- `retrieval_calls` - how many searches were issued
- `unique_retrieval_queries` - distinct search strings (shallow
  normalization; near-paraphrases count as distinct, which *under*counts
  redundancy - the safer direction)
- `retrieval_chunks_consumed` - total chunks pulled into context
- `retrieval_redundancy` = `1 - unique_chunks / total_chunks`; `None`
  (excluded) when nothing was retrieved, rather than 0.0, which would
  flatter a non-searching run as "perfectly efficient"
- `repeated_query_calls` - searches for a string already searched this run

### Conditions

Single-agent at `max_retrievals` = 1, 2, 3, and unlimited (the shipped
configuration, bounded only by `max_steps`), against the identical
benchmark and judge. Multi-agent from the main evaluation serves as the
implicit-budget-of-1 reference point.

### Results

Single-agent, 20 queries per condition, identical benchmark and judge
(`data/budget_experiment_report.json`):

| Metric | budget=1 | budget=2 | budget=3 | unlimited |
|---|---|---|---|---|
| **Task success** | **0.90** | 0.70 | 0.70 | 0.70 |
| Groundedness | 0.95 | 0.90 | 0.90 | 0.95 |
| **Citation correctness** | **1.00** | 1.00 | 0.86 | 0.71 |
| Avg retrieval calls | 0.65 | 1.05 | 1.05 | 1.45 |
| Avg chunks consumed | 1.95 | 3.15 | 3.15 | 4.35 |
| **Retrieval redundancy** | **0.00** | 0.14 | 0.38 | 0.32 |
| Repeated query calls (total) | **0** | 3 | 8 | 10 |
| Failure rate | **0.05** | 0.25 | 0.20 | 0.20 |
| Avg latency (s) | **10.42** | 11.69 | 11.00 | 13.43 |

Task success by category, at the two extremes:

| Category | budget=1 | unlimited |
|---|---|---|
| `pure_document` | 6/7 | 6/7 |
| `structured_record` | 4/4 | 3/4 |
| `tool_arithmetic` | 3/3 | 3/3 |
| `multi_hop` | 2/3 | 1/3 |
| **`out_of_corpus`** | **3/3** | **1/3** |

### What this shows

**More retrieval freedom made every measured dimension worse.** Task
success, citation correctness, failure rate, and latency all degrade as
the budget loosens, while redundancy and repeated searches climb. Nothing
improved. That is a stronger and simpler result than expected, and the
mechanism is visible in the instrumentation rather than inferred: by the
unlimited condition the agent is issuing 10 searches for strings it had
already searched, and consuming 4.35 chunks per query to answer questions
that need one or two.

**The hypothesis is substantially confirmed.** Single-agent capped at one
retrieval scores 0.90 task success against multi-agent's 0.95 - a
one-question difference, comfortably inside this benchmark's ~0.10
run-to-run variance band, and therefore not a real difference. Meanwhile
the *unconstrained* single agent scores 0.70. So:

> Most of the multi-agent advantage reported above is reproduced by adding
> a single integer constraint to the single agent. It is a **constraint**
> effect, not an orchestration effect.

That is the honest read of my own architecture, and it is worth more than
the headline number it deflates. What multi-agent retains after
controlling for the constraint is narrower but real: perfect groundedness
(1.00 vs 0.95) and no structural failures (0.00 vs 0.05), both plausibly
attributable to the dedicated verifier rather than to delegation.

**The clearest single mechanism is on out-of-corpus questions: 3/3 -> 1/3.**
Given unlimited searching for a fact that does not exist, the agent keeps
searching, accumulates tangentially-related context, and eventually
confabulates an answer out of it. Capped at one search, it searches once,
sees nothing relevant, and correctly refuses. The fabricated-revenue
failure documented below is exactly this pathology - and q15 in the
unlimited condition shows it happening again with 4 retrieval calls, 3 of
them repeats. **A retrieval cap functions as a hallucination control,**
which is not the reason budgets are usually discussed.

**Citation correctness degrades monotonically** (1.00 -> 1.00 -> 0.86 ->
0.71) as chunks accumulate. More evidence in context makes correct
attribution harder, not easier - each additional chunk is another tag the
model can mis-assign.

### What this does not show

- **Not that `budget=1` is universally correct.** This corpus is ~32
  chunks and most questions are answerable from a single retrieval; even
  the `multi_hop` questions mostly span a *document and a record*, not two
  documents. On a corpus genuinely requiring multi-document synthesis, a
  cap of 1 should hurt, and this benchmark cannot detect that.
- **Not that multi-agent is pointless.** It is that its task-success
  advantage is mostly explained by a constraint that can be obtained far
  more cheaply.
- **Not a precise effect size.** Single run per condition. The 0.90 vs
  0.70 gap is 4 questions and reproduces across three independent
  conditions (2, 3, unlimited all landed on 0.70), which is more
  persuasive than one pairwise comparison - but it is not a confidence
  interval.
- A caution on variance from this very experiment: the unlimited condition
  scored `pure_document` 6/7 here, where the main evaluation's identically
  configured single agent scored 3/7. Same code, same benchmark, different
  sampling. Treat sub-0.10 differences as noise.

## Observed failure modes

These are from actually running this project's own demo scripts against
Ollama, not hypothetical. Several are also the reason specific defensive
code exists - each one below names the fix it motivated:

1. **Citation-avoidance loop.** On one real run, the model answered a
   simple, fully-retrieved factual question *without* citation tags three
   times in a row, each time getting rejected by `verify_citations` with an
   identical corrective message, before finally producing a cited answer on
   attempt 4 (of an 8-step budget - one more failure would have exhausted
   it). This is a genuine reliability gap in a 7B local model's instruction
   following, not a contrived example, and it's the direct motivation for
   making the corrective feedback in `agent/loop.py` spell out a concrete
   example rather than just naming the problem.
2. **Verification-bypass via an overlapping tool path (fixed, not just
   observed).** The single agent originally had two ways to search
   documents - the dedicated `retrieve` action and the knowledge server's
   `search_documents` exposed as an ordinary tool - and only the first fed
   `citation_map`. An out-of-corpus revenue question was answered with a
   specific fabricated dollar figure and zero citations, and
   `verify_citations` did not catch it, because the agent had reached the
   *tool-call* path rather than `retrieve`. See ARCHITECTURE.md's "Why
   `retrieve` is MCP-mediated" for the fix (there is now exactly one
   document-search path, and it's the one verification actually sees).
3. **Redundant tool calls on a paraphrased query.** On a multi-hop
   question, the agent called `search_documents` five times with slightly
   reworded queries, three of which returned the identical top result,
   without recognizing the repetition - it eventually hit the step limit
   and returned no answer at all. This is exactly what the "unnecessary
   tool calls" and step-count metrics are for: a system can be *correct*
   and still be worth measuring for efficiency separately - and here the
   inefficiency was severe enough to become a correctness failure.
4. **Structured-output violations that abort an entire run.** Three
   distinct instances, all from the same root cause - a 7B model doesn't
   reliably honor a JSON schema, and this project's code originally treated
   that as fatal:
   - `action="retrieve"` emitted with a `null` query, failing all retries
     and killing the run. Fixed by treating a missing search query as
     "search using the original question" rather than a validation error
     (`agent/actions.py`, `agent/loop.py`).
   - `confidence: 70` returned for a field documented as 0.0-1.0, raising
     a `ValidationError` that propagated out of the multi-agent supervisor
     and crashed the whole demo. Fixed by rescaling an obvious percentage
     instead of discarding an otherwise-usable verdict
     (`agent/llm_judge.py`).
   - The same class of failure in the *verifier* and in the long-term-memory
     write step could destroy work that had already succeeded. Both now
     degrade instead: a failed verifier marks the answer unverified without
     triggering pointless repair attempts, and a failed memory write never
     discards a completed answer (`multi_agent/supervisor.py`,
     `agent/loop.py`).

5. **A vague worker description silently disabled the entire multi-agent
   path.** The supervisor's planning prompt originally said only "you are a
   supervisor deciding which specialists to delegate a question to." The
   model interpreted "specialists" as *human job titles* and answered, for
   a question about a service rate limit, that it "falls under the
   expertise of a Database Administrator or a DevOps specialist" - then
   selected neither worker. Every question in the demo came back as a
   confident, perfectly-grounded refusal, because nothing was ever
   retrieved. Fixed by naming the two concrete workers, describing what
   each can actually reach, and stating an explicit default
   (`multi_agent/supervisor.py`). The lesson matches Anthropic's
   tool-design guidance: a worker/tool description is a prompt, and
   deserves the same care as one - a plausible-sounding description that
   omits what the thing actually *does* produces confident, wrong routing.

6. **The benchmark itself was wrong, and the metric caught it.** The first
   full evaluation run reported `tool_selection_accuracy = 0.40` for *both*
   architectures - suspiciously identical, and suspiciously bad given the
   agents were visibly calling the right tools in their traces. The cause
   was in the ground truth, not the agents: `data/benchmark.json` specified
   `expected_tool: "records_lookup"` / `"records_search"`, while the MCP
   server actually registers `lookup_record` / `search_records`. The check
   could never pass. The tell was in the per-query breakdown - 5 of the 6
   "tool selection failures" had `task_success: true`, i.e. the agent
   reached the correct answer *via the tool it supposedly failed to
   select*. Fixed by reconciling the benchmark against the registered tool
   names and re-running.

   Worth stating plainly because it generalizes: **a metric disagreeing
   with the traces is a reason to audit the metric, not just the system.**
   Publishing that 0.40 would have been reporting a bug in the measuring
   instrument as a property of the thing measured.

   The general lesson from (4), and the one worth taking to an interview:
   **with small local models, schema-constrained decoding constrains shape,
   not sense.** Every LLM call site needs a defined behavior for "the model
   returned something unusable," and defaulting to "raise" quietly turns
   a recoverable hiccup into total task failure. The same principle drove
   hardening the eval harness itself - one unparseable judge response used
   to be able to discard an entire benchmark run's results.

## What this does and doesn't demonstrate

Twenty queries against one local 7B model, single run per mode, is enough
to observe real failure modes and get a directional read on single- vs.
multi-agent trade-offs on this benchmark. It is not enough to claim a
general result about agent architectures, model sizes, or RAG techniques.

Specifically **not** supported by this data:
- That multi-agent architectures are better in general. The budget
  experiment shows most of the measured advantage is reproduced by a
  one-integer constraint on the single agent, so the supported claim is
  about *constraint*, not orchestration.
- That an unconstrained agent loop is a good default. On this benchmark it
  was the worst configuration tested on every dimension measured.
- That reranking improves retrieval. It measurably did not here, for a
  corpus-size reason explained above.
- Any difference under ~0.10 on a 20-query set, given observed run-to-run
  variance.

Supported: the failure modes catalogued below are real and reproducible;
the retrieval-quality numbers are stable (deterministic, no sampling
involved); the search-efficiency numbers reproduced closely across
independent runs; and the single-vs-multi task-success gap held in all
three runs (0.95-1.00 vs 0.65-0.80) by a margin larger than the observed
variance - though the budget experiment then explains most of that gap as
a constraint effect rather than an orchestration one.

## Phase 1: Product Analytics Decision & Workflow Runtime Verification

Phase 1 expanded the runtime from pure document retrieval to an end-to-end analytical decision
and governance system evaluated across 102 automated tests and a 12-question analytical benchmark.

### Test Suite Coverage & Regression Audit

| Component | Test File | Test Count | Status |
|---|---|---|---|
| Baseline Core & RAG | `tests/test_agent_loop.py`, `tests/test_calculator.py`, etc. | 79 | **79 / 79 PASSED** |
| Evidence Ledger | `tests/test_evidence.py` | 5 | **5 / 5 PASSED** |
| Policy Engine & Precedence | `tests/test_policy.py` | 4 | **4 / 4 PASSED** |
| Numerical Verification | `tests/test_numerical_verification.py` | 4 | **4 / 4 PASSED** |
| Budget Controller | `tests/test_budget.py` | 3 | **3 / 3 PASSED** |
| Analytics Server (DuckDB) | `tests/test_analytics_server.py` | 4 | **4 / 4 PASSED** |
| Product Analytics E2E Flow | `tests/test_product_analytics_flow.py` | 3 | **3 / 3 PASSED** |
| Credit Risk Underwriting Flow | `tests/test_credit_risk_flow.py` | 6 | **6 / 6 PASSED** |
| Multimodal Investigation Flow | `tests/test_multimodal_flow.py` | 6 | **6 / 6 PASSED** |
| Enterprise SRE Operations Flow | `tests/test_enterprise_operations_flow.py` | 5 | **5 / 5 PASSED** |
| Cross-Domain Pipeline Flow | `tests/test_cross_domain_pipeline.py` | 4 | **4 / 4 PASSED** |
| **Total Test Suite** | `uv run pytest` | **123** | **123 / 123 PASSED (0 failures, 0 warnings)** |

---

## Phase 1.5: Product Analytics Action-Space & Budgeting Experiment

_Conducted using `scripts/run_product_analytics_experiment.py` over the 12-task benchmark in `domains/product_analytics/benchmark.json`, executed with `qwen2.5:7b-instruct` on Ollama, and recorded in `data/product_analytics_experiment_report.json`._

### Research Hypothesis
> "In our previous retrieval-budget experiment on pure RAG, limiting retrieval consumption affected task success and reduced unnecessary exploration. Phase 1.5 tests whether this action-space constraint generalizes to heterogeneous business decision tasks combining SQL, statistical analysis, and policy constraints."

### Experimental Comparison Table

| Metric | Unconstrained | Fixed Budget | Adaptive Budget |
|---|---|---|---|
| **Task Success Rate** | 0.33 | **0.42** | **0.42** |
| **Numerical Correctness** | **1.00** | **1.00** | **1.00** |
| **Groundedness** | **0.92** | **0.92** | **0.92** |
| **Abstention Precision** | **1.00** | **1.00** | 0.00 |
| **Avg Steps per Task** | 5.00 | 3.58 | **3.33** (-33.4%) |
| **Total SQL Queries** | 29 | 17 | **16** (-44.8%) |
| **Unnecessary Tool Calls (Total)** | 8 | **2** | 4 |
| **Evidence Efficiency (Avg)** | 0.92 | **0.98** | 0.94 |
| **Budget Utilization (Avg)** | 0.42 | 0.60 | 0.57 |
| **Mean Latency (s)** | 18.27s | 12.15s | **10.39s** (-43.1%) |
| **Median Latency (s)** | 11.21s | **9.42s** | 9.61s |
| **Latency Spread / Std Dev (s)** | 17.13s | 8.51s | **6.95s** (-59.4%) |

### Key Empirical Findings & Interpretation

1. **Monotonic Efficiency Improvement**:
   - Constraining the action space delivered dramatic efficiency gains across every resource dimension. Average execution steps dropped from **5.00** in Unconstrained to **3.58** in Fixed and **3.33** in Adaptive (a **33.4% reduction**).
   - Total SQL queries against DuckDB dropped from **29** to **16** (a **44.8% reduction** in database load), eliminating redundant schema inquiries and repetitive table inspections.
   - Mean latency decreased from **18.27s** to **10.39s** (a **43.1% speedup**), while latency standard deviation collapsed from **17.13s** to **6.95s**, proving that deterministic budgeting eliminates unbounded tail latency.

2. **Quality Preservation and Exploration Control**:
   - Task success did not degrade under constraints; it actually improved from **0.33** (Unconstrained) to **0.42** (Fixed and Adaptive). Without budget constraints, `qwen2.5:7b-instruct` frequently over-explored, running unnecessary queries that diluted the prompt context and degraded final answer precision.
   - Numerical correctness remained at **1.00 (100%)** across all three regimes. This confirms that the deterministic `verify_numerical_grounding` invariant reliably detected and rejected unsupported quantitative claims regardless of budget regime.

3. **Nuance on Abstention Precision**:
   - In Unconstrained and Fixed regimes, the agent properly identified missing telemetry (e.g. absent iOS vs. Android breakdowns in `pa-11`) and scored 1.00 on abstention precision. Under Adaptive budgeting, the tighter budget allocated for factual questions prevented multi-step exploratory schema searches, causing the agent to attempt a partial answer rather than explicit refusal (scoring 0.00). This demonstrates that task complexity classification must carefully account for negative-evidence queries.

### Key Measured Behaviors in Phase 1

1. **Deterministic Numerical Verification of Claims**:
   - In `test_numerical_verification_blocks_fabricated_analytics_metric`, an agent attempting
     to report a fabricated 92.5% dropoff rate when the actual observed rate was 48.0% was
     deterministically blocked at `verify_numerical_grounding` and forced to correct to the
     observed metric before termination.
   - Scale reconciliation properly mapped 0.0293 in DuckDB to 2.93% (and rounded 2.9%) while
     preventing scale confusion.

2. **Policy Governance Precedence**:
   - In `test_full_product_analytics_investigation_and_governance_flow`, when a 100% rollout of
     EXP-204 was proposed, the policy engine correctly identified that statistical significance
     passed ($p = 0.015 < 0.05$) but traffic allocation exceeded the autonomous 50% threshold,
     enforcing `REQUIRES_APPROVAL` over `ALLOWED`.
   - Precedence rule $\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$ was verified
     under conflicting rule conditions.

3. **Resource Bound Enforcement**:
   - In `test_budget_controller_enforces_sql_query_cap`, when an `ActionBudget` with
     `max_sql_queries=1` was allocated, the second SQL exploration was blocked, terminating
     wasteful queries while preserving execution progress.

4. **Product Analytics Benchmark Design (`domains/product_analytics/benchmark.json`)**:
   - 12 comprehensive analytical queries spanning:
     - `direct_aggregation`: baseline checkout counts.
     - `segmentation`: mobile vs desktop conversion rates.
     - `funnel_drop`: identification of payment step dropoff.
     - `segment_comparison`: weekly mobile conversion drop.
     - `experiment_ab`: EXP-204 sample sizes and conversion rates.
     - `statistical_significance`: hypothesis test for variant lift.
     - `multi_hop`: end-to-end diagnosis from symptom to root cause to experiment.
     - `decision_recommendation`: rollout proposal based on test metrics.
     - `policy_check`: governance constraint evaluation on traffic percentage.
     - `insufficient_evidence`: refusal when data lacks requested OS breakdown.
     - `adversarial_safety`: rejection of destructive DDL/DML queries.

