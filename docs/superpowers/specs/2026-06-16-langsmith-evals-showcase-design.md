# LangSmith Evals Showcase — Design Spec

**Date:** 2026-06-16
**Author:** João Pedro Gomes Ferreira (GitHub: Jpgomesf)
**Status:** Approved design (produced from brainstorming session)

## Summary

`langsmith-evals-showcase` is a public portfolio repository whose single purpose is to demonstrate end-to-end mastery of LangSmith's evaluation capabilities. It is structured as an "eval gym": five small, independent apps-under-test, each deliberately chosen to spotlight a distinct family of LangSmith eval features — offline `evaluate()` with heuristic and summary evaluators, custom code evaluators emitting multiple feedback keys, retrieval metrics plus LLM-as-judge, agent trajectory evaluation, Prompt Hub versioning, and pairwise comparison. Three cross-cutting tracks (online evals with an automation rule, an SDK-created annotation queue, and a pytest-based CI regression gate) round out the surface area. The apps are intentionally trivial; the eval craft is the product. Display is the LangSmith UI itself — the README narrates results with screenshots and deep-links rather than a bespoke dashboard.

## Goals

- Demonstrate, in working code, the breadth of LangSmith eval primitives: datasets/splits, offline `evaluate()`/`aevaluate()`, heuristic and summary evaluators, custom code evaluators, LLM-as-judge, trajectory evaluation, `evaluate_comparative()`, Prompt Hub pull/version, online evals, annotation queues, and the pytest testing integration.
- Keep each scenario small, readable, and independently runnable so a reviewer can read any one folder top-to-bottom in minutes.
- Be provider-agnostic: both apps-under-test and judges swap models via a single env var.
- Be reproducible: committed datasets, local embeddings (no embeddings API key), idempotent seeding, pinned dependencies.
- Make the eval craft legible: clear evaluator contracts, structured grading outputs, and a strategy guide cross-linked from the README.

## Non-goals

- **No custom dashboard or report generator.** The LangSmith UI is the display surface. The repo ships no HTML reports, no plotting layer, no results server. The README links to live experiments and embeds screenshots.
- **Not wrapping an existing application.** The apps-under-test are purpose-built minimal targets, not instrumentation bolted onto a real product.
- **Not a tutorial-notebook format.** No Jupyter notebooks. The deliverable is a clean `uv`-managed `src`-layout package with a typer CLI and pytest suite — engineered, not narrated cell-by-cell.

## Decisions (locked)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Provider-agnostic via `langchain.init_chat_model` with config strings like `anthropic:claude-...` / `openai:gpt-...`, swappable by `MODEL` / `JUDGE_MODEL`. | One knob to re-run any scenario on any provider; no per-scenario model wiring; traces flow to LangSmith natively. |
| D2 | Display is the LangSmith UI; README narrates with screenshots + deep-links. | Keeps the codebase focused on eval craft, not reinventing a reporting tool LangSmith already provides. |
| D3 | LLM-judges are custom (`init_chat_model` + structured Pydantic grade), not `openevals`. | Shows the actual mechanics of judge construction; `openevals` is mentioned in the strategy guide as an alternative but is not a dependency. |
| D4 | RAG embeddings use `fastembed` locally (cosine over a small committed corpus). | Removes the need for an embeddings API key; makes retrieval deterministic and reproducible. |
| D5 | `uv`-managed, `src` layout, package `src/evals_showcase/`. | Standard, clean Python packaging; clear import boundaries; easy CI. |
| D6 | Datasets are committed as JSONL and seeded idempotently: dataset existence is guarded by name; examples carry stable deterministic IDs so re-seeding upserts rather than duplicates. | Reproducible eval sets that version with the repo; re-running `seed` is safe. |
| D7 | A reusable evaluators library (`evaluators/heuristic.py`, `judges.py`, `summary.py`, `trajectory.py`) shared across scenarios. | DRY; one place to read every evaluator contract; scenarios compose from it. |
| D8 | A single typer CLI (`seed | run <scenario> | run-all | pairwise | online | annotate`) is the entry point for every operation. | One discoverable surface; no scattered `__main__` scripts. |
| D9 | Online-eval automation rule is created in the LangSmith UI; everything else (traffic, feedback, queue) is SDK-driven. | The automation/rules tab cannot be fully scripted in the pinned SDK; the split is documented honestly with screenshots and a click-path. |
| D10 | Pinned stack: `langsmith==0.3.45`, `langchain==0.3.25`, `langchain-core==0.3.84`, `langgraph==1.1.6`, `langchain-openai==1.1.13`, `pydantic==2.12.5`, `pydantic-settings==2.13.1`, `pytest==8.4.2`; to add `langchain-anthropic`, `fastembed`, `typer`, `ruff`, `mypy`. | The API reference is authoritative for these versions only; behavior changed across the 0.2 → 0.3 line. |

## Architecture

### Provider-agnostic model layer

`models.py` exposes two factories built on `langchain.init_chat_model`:

- `get_chat_model()` — the model for apps-under-test, configured from `settings.model` (e.g. `"anthropic:claude-..."`).
- `get_judge_model()` — the model for LLM-judge evaluators, configured from `settings.judge_model`.

Both return a `BaseChatModel`. Apps and judges use `.with_structured_output(SomePydanticModel)` for structured generation and grading, respectively. Provider keys (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) are consumed by `init_chat_model`, not by LangSmith. Because tracing is native, every model call inside a `@traceable` target or an `evaluate()` run is captured automatically when `LANGSMITH_TRACING=true`.

### Configuration

`config.py` uses `pydantic-settings` to load a `Settings` object from environment / `.env`:

- LangSmith: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` (legacy `LANGCHAIN_*` aliases honored by the SDK's namespace fallback).
- Models: `MODEL`, `JUDGE_MODEL`.
- Provider keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (only the one matching the chosen provider is required).

`Settings` is the single source of truth; nothing reads `os.environ` directly. `.env.example` documents every variable.

### Eval-gym structure

```
src/evals_showcase/
  config.py            # pydantic-settings Settings
  models.py            # get_chat_model / get_judge_model (init_chat_model)
  embeddings.py        # fastembed local embedding + cosine retriever helper
  datasets.py          # idempotent create/version/split via Client
  cli.py               # typer: seed | run | run-all | pairwise | online | annotate
  online.py            # traffic simulator + feedback emission
  annotation.py        # annotation-queue create + enqueue
  evaluators/
    heuristic.py       # exact-match, recall@k, numeric-tolerance, field-match
    judges.py          # LLM-as-judge evaluators (structured Pydantic grades)
    summary.py         # macro-F1, precision/recall, confusion matrix
    trajectory.py      # tool-sequence / tool-selection evaluators
  scenarios/
    classify/   { app.py, data.jsonl, experiment.py }
    extract/    { app.py, data.jsonl, experiment.py }
    rag/        { app.py, data.jsonl, corpus.jsonl, experiment.py }
    agent/      { app.py, data.jsonl, experiment.py }
    generate/   { app.py, data.jsonl, experiment.py }
tests/                 # @pytest.mark.langsmith gates + pure-unit evaluator tests
docs/                  # strategy guide, screenshots, deep-links
```

Each scenario folder is self-contained: `app.py` is the app-under-test (a plain `inputs: dict -> outputs: dict` callable or async callable), `data.jsonl` is the committed dataset, and `experiment.py` wires the dataset to its evaluators via `evaluate()` / `aevaluate()` / `evaluate_comparative()`.

### Dataset seeding strategy

Datasets are committed as JSONL files (one example per line: `{"inputs": {...}, "outputs": {...}, "split": [...]}`) and loaded by `datasets.py` so that re-running `seed` is safe:

1. **Guard the dataset by name.** `create_dataset` raises on a duplicate name in 0.3.45, so seeding first checks `client.has_dataset(dataset_name=...)` and calls `client.create_dataset(...)` only when absent.
2. **Upsert examples with stable IDs.** Bulk-upsert with `client.create_examples(dataset_name=..., examples=[...])` using the keyword-only `examples=` shape required by 0.3.45. Each example carries a stable, deterministic `id` (a UUIDv5 derived from `dataset_name` + a per-line key) so re-seeding upserts the same rows instead of appending duplicates — `create_examples` returns an `UpsertExamplesResponse`. Without explicit IDs the call would append new examples on every run.
3. **Tag splits per example** via `split: str | list[str]` (e.g. `["train"]` / `["test"]`); split-scoped runs read them with `client.list_examples(dataset_name=..., splits=["test"])`.

The committed JSONL is the source of truth, so the eval set versions alongside the code. For reproducible pinned runs, experiments may pin a dataset version at read time with `list_examples(..., as_of=<version>)`.

### Reusable evaluators library

All evaluators follow the 0.3.45 name-resolved argument contract (params chosen from `run, example, inputs, outputs, reference_outputs, attachments`; for comparative, `runs, example, inputs, outputs, reference_outputs`; for summary, `runs, examples, inputs, outputs, reference_outputs`). Return-value coercion: `bool|int|float -> {"score": x}`, `str -> {"value": x}`, `list[dict] -> {"results": [...]}` (multiple feedback keys), `dict`/`EvaluationResult` passthrough; empty/falsy raises.

- **`heuristic.py`** (row-level) — `exact_match(outputs, reference_outputs)`; `recall_at_k(outputs, reference_outputs)`; numeric-tolerance match; normalized per-field match. Single- or multi-key dicts.
- **`judges.py`** (row-level LLM-as-judge) — each calls `get_judge_model().with_structured_output(Grade).invoke(prompt)`, then maps the structured grade to `{"key", "score"|"value", "comment"}`. `value` carries categorical verdicts; `comment` carries the rationale; `feedback_config` adds UI range hints for continuous scores. Examples: faithfulness/groundedness, answer-correctness, context-relevance, multi-criteria rubric.
- **`summary.py`** (experiment-level) — receives sequences (`outputs=[run.outputs, ...]`, `reference_outputs=[example.outputs, ...]`); computes macro-F1, per-class precision/recall, and a confusion matrix. Returns a single dict / `EvaluationResults`.
- **`trajectory.py`** (row-level, run-tree) — takes `run` and walks the run tree to extract the tool-call sequence; scores tool-sequence match and tool-selection correctness.

### Typer CLI

`cli.py` exposes one command group:

| Command | Action |
|---------|--------|
| `seed` | Idempotently create/update every dataset from committed JSONL (optionally `--scenario <name>`). |
| `run <scenario>` | Run that scenario's offline experiment (`evaluate`/`aevaluate`) with its evaluators + summary evaluators. Flags: `--split`, `--repetitions`, `--max-concurrency`. |
| `run-all` | Run every scenario's offline experiment in sequence. |
| `pairwise` | Run the `generate` `evaluate_comparative()` across two prompt-hub versions. |
| `online` | Start the production-traffic simulator: emit `@traceable` runs into `LANGSMITH_PROJECT` and attach feedback. |
| `annotate` | Create the annotation queue and enqueue a sampled subset of runs via the SDK. |

## Scenarios

### 1. classify — intent/category classifier

- **App-under-test:** a single structured-output LLM call. `app(inputs)` takes `{"text": "..."}` and returns `{"label": "<category>"}` via `get_chat_model().with_structured_output(IntentLabel)`.
- **Dataset shape:** `inputs={"text": str}`, `outputs={"label": str}`, with `split` tags (`train`/`test`). A handful of intents (e.g. `billing`, `cancellation`, `technical`, `general`).
- **Evaluators:** row-level `exact_match` (heuristic); summary-level `macro_f1`, per-class `precision`/`recall`, and `confusion_matrix` from `summary.py`.
- **LangSmith features demonstrated:** datasets + splits; offline `evaluate()`; heuristic exact-match evaluator; **summary evaluators** (macro-F1, precision/recall, confusion matrix); `num_repetitions` for variance/noise reporting; and the pytest CI regression gate (this scenario anchors the gate).

### 2. extract — text → JSON field extraction

- **App-under-test:** transaction-style extraction. `app(inputs)` takes `{"text": "..."}` and returns a dict serialized from a Pydantic model (e.g. `Transaction(merchant, amount, date)`), produced via `.with_structured_output(Transaction)`.
- **Dataset shape:** `inputs={"text": str}`, `outputs={"merchant": str, "amount": number, "date": str}` reference fields.
- **Evaluators:** a single **custom code evaluator** returning a `list[dict]` → multiple feedback keys in one pass: `json-validity` (structural), per-field normalized match (`field:merchant`, `field:date`), numeric-tolerance match (`field:amount`), and an aggregate `field-accuracy`. Built from `heuristic.py` helpers.
- **LangSmith features demonstrated:** custom **code evaluators emitting multiple feedback keys** from one function (the `list[dict] -> {"results": [...]}` coercion path); numeric-tolerance and normalized-string matching; aggregate accuracy as a derived feedback key.

### 3. rag — tiny in-memory retriever + answer generation

- **App-under-test:** local `fastembed` embeddings + cosine similarity over a small committed corpus (`corpus.jsonl`) retrieves top-k passages; `get_chat_model()` then generates an answer grounded in the retrieved context. `app(inputs)` returns `{"answer": str, "context": list[str]}` so evaluators can inspect the intermediate retrieved context.
- **Dataset shape:** `inputs={"question": str}`, `outputs={"answer": str, "relevant_ids": list[str]}` (reference answer + the gold passage ids for retrieval scoring).
- **Evaluators:** heuristic `recall_at_k` over `relevant_ids` (retrieval recall); LLM-judges from `judges.py` for **faithfulness/groundedness** (answer supported by `context`), **answer-correctness** (reference-based, against the gold answer), and **context-relevance** (retrieved passages relevant to the question).
- **LangSmith features demonstrated:** evaluating an **intermediate retrieved-context** value (not just the final answer); heuristic retrieval recall@k; reference-based and reference-free LLM-as-judge; local reproducible embeddings via `fastembed`.

### 4. agent — small LangGraph ReAct agent

- **App-under-test:** a LangGraph ReAct agent with 2–3 tools (e.g. a calculator, a lookup, a unit converter). Run via `aevaluate()` since the agent is async. `app(inputs)` returns the final answer; the full run tree (with tool calls) is captured by tracing.
- **Dataset shape:** `inputs={"task": str}`, `outputs={"answer": str, "expected_tools": list[str]}` (reference final answer + expected tool sequence).
- **Evaluators:** `trajectory.py` evaluators that take `run`, walk the run tree to extract the actual tool sequence, and score **trajectory match** (tool sequence vs `expected_tools`) and **tool-selection correctness**; plus a `judges.py` **final-answer judge** for correctness.
- **LangSmith features demonstrated:** **trajectory evaluation** over the run tree; tool-selection correctness; final-answer LLM-judge; `aevaluate()` with an async target.

### 5. generate — summarize/draft with Prompt Hub versioning

- **App-under-test:** a draft/summary generator whose prompt is **pulled from LangSmith Prompt Hub** (`client.pull_prompt("<owner>/summarize:<tag>")`), composed as `prompt | get_chat_model()`. Two prompt versions are pushed (`push_prompt`) so the scenario can compare them.
- **Dataset shape:** `inputs={"document": str}`, `outputs` may be empty or hold a reference summary; the primary signal is judge-based and pairwise.
- **Evaluators:** a reference-free **multi-criteria LLM-judge rubric** (categorical + continuous feedback — e.g. faithfulness verdict + a continuous quality score with `feedback_config`); plus a **comparative evaluator** returning `{"key", "scores": {run_id: score}}` for `evaluate_comparative()` across the two prompt versions.
- **LangSmith features demonstrated:** **Prompt Hub versioning** (`pull_prompt` with `:tag`/commit pinning, `push_prompt`); reference-free multi-criteria rubric judge (mixed categorical + continuous feedback); **PAIRWISE comparison** via `evaluate_comparative()` (synchronous-only in 0.3.45; `randomize_order=True` to debias position-sensitive judges) and the LangSmith comparison view.

## Cross-cutting tracks

### Online evals (production-traffic simulator + automation rule)

- **SDK side (`online.py`):** wrap an app with `@traceable` (imported from the `langsmith` top level) and replay a stream of synthetic production inputs so runs land in `LANGSMITH_PROJECT`. Feedback may be attached programmatically with `client.create_feedback(run_id, key=..., score=..., value=..., comment=...)`.
- **UI side:** the **online evaluator / automation rule** — sample N% of incoming runs, auto-run an LLM-judge — is configured on the project's **Rules / Automations** tab. This cannot be fully scripted in the pinned SDK.
- **Documentation:** `docs/` captures the exact click-path to create the rule plus screenshots and a deep-link, with an explicit note on the SDK-vs-UI split (decision D9).

### Annotation queue (human grading)

- **SDK create + enqueue (`annotation.py`):** `client.create_annotation_queue(name=..., rubric_instructions=...)` then sample runs (`client.list_runs(project_name=..., limit=...)` + `random.sample`) and `client.add_runs_to_annotation_queue(queue_id, run_ids=[...])`.
- **Human grading happens in the LangSmith UI:** reviewers apply the rubric to the enqueued subset. The repo documents the queue and rubric; it does not attempt to script the human grading step.

### CI regression gate (GitHub Actions)

- A GitHub Actions job runs `pytest` over the `@pytest.mark.langsmith` tests. The plugin auto-registers via the `pytest11` entry point (`langsmith_plugin -> langsmith.pytest_plugin`); no explicit import or `-p` flag is needed.
- Each gate logs inputs/outputs/reference via `langsmith.testing` (`t.log_inputs`, `t.log_outputs`, `t.log_reference_outputs`, `t.log_feedback`) and asserts a **score threshold** (e.g. `exact_match == 1` per example, or an aggregate floor) so standard pytest pass/fail gates the job while results also stream to LangSmith as a test experiment.
- **Required secrets:** `LANGSMITH_API_KEY` and one provider key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`), matching the configured `MODEL`/`JUDGE_MODEL`.
- **Optional nightly:** a scheduled workflow runs the full offline experiments (`run-all`, and `pairwise`) against the live datasets to catch drift and refresh experiment history; this is separate from the per-PR gate.

## Testing strategy

- **Pure-unit, deterministic evaluator tests (no API):** the heuristic/summary/trajectory evaluators are pure functions over dicts/lists and are tested offline with `LANGSMITH_TRACING=false` and no `@pytest.mark.langsmith`. These assert exact numeric behavior (e.g. macro-F1 on a known confusion matrix, recall@k boundaries, multi-key field accuracy, trajectory match) and run in CI with zero network and zero cost. They are the primary regression safety net for eval logic.
- **LangSmith pytest gates (with thresholds):** `@pytest.mark.langsmith` tests exercise an app-under-test end-to-end, log via `langsmith.testing`, and assert a score threshold. These require the LangSmith and provider secrets and run in the gated CI job (and optionally the nightly).
- LLM-judge evaluators themselves are exercised in the gated/nightly paths, not the pure-unit path, because they are nondeterministic and incur cost.

## Reproducibility & limitations

- **Required credentials:** `LANGSMITH_API_KEY` plus one provider key. No embeddings key is needed — `fastembed` runs locally (D4), which keeps RAG retrieval deterministic and reproducible.
- **Committed, idempotent datasets:** JSONL data files version with the repo and seed idempotently, so any reviewer can reproduce the exact eval sets.
- **LLM-judge nondeterminism and cost:** judge outputs vary run-to-run and cost money. Mitigations: pinned `JUDGE_MODEL`, structured Pydantic grades (constrained outputs reduce variance), `num_repetitions` to quantify noise where it matters, keeping datasets small, and `randomize_order` for pairwise debiasing. The trade-offs and recommended practices are documented in the strategy guide, cross-linked from the README.
- **Online-eval rule creation is partly manual:** the automation rule is created in the UI (D9); the repo documents the click-path with screenshots rather than claiming full automation.
- **Provider parity:** results differ across providers; experiment metadata records `model`/`judge_model` so cross-provider runs stay comparable in the UI.

## Build order / milestones

Each scenario is an independently shippable milestone — it can be merged and demoed on its own once its dataset, app, evaluators, and experiment wiring pass.

1. **Foundation** — `uv` project + `src` layout, `config.py`, `models.py`, `embeddings.py`, `datasets.py`, `evaluators/` skeleton, `cli.py` skeleton, tooling (ruff, mypy, pytest, Makefile, `.env.example`, GitHub Actions scaffold). Pure-unit evaluator tests for the heuristic/summary helpers.
2. **classify** — dataset + splits, app, exact-match + summary evaluators, offline experiment, and the first pytest CI regression gate.
3. **extract** — dataset, app, the multi-key custom code evaluator, offline experiment, unit tests for field accuracy.
4. **rag** — committed corpus + `fastembed` retriever, app, recall@k + faithfulness/correctness/relevance judges, offline experiment.
5. **agent** — LangGraph ReAct agent with tools, trajectory evaluators, final-answer judge, async `aevaluate()` experiment.
6. **generate** — Prompt Hub push of two versions, rubric judge, comparative evaluator, `evaluate_comparative()` pairwise run.
7. **Cross-cutting** — online traffic simulator + documented automation rule; annotation queue create + enqueue; nightly full-experiment workflow.
8. **README polish & screenshots** — narrate every scenario and track with LangSmith UI screenshots and deep-links; finalize the strategy guide cross-links.
