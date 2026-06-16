# LangSmith API Reference

Concise, version-grounded cheat-sheet of the exact LangSmith evaluation APIs this repository uses. Every signature below was confirmed by Python introspection against the **installed** `langsmith==0.8.16`. Where behavior is version-sensitive it is flagged inline. Do not assume newer-docs behavior (e.g. `aevaluate_comparative`, openevals helpers) applies here.

> Ground-truth note: this repo pins `langsmith==0.8.16`, `langchain==1.3.9`, `langgraph==1.2.5`. Signatures and the evaluator callable contract have shifted across major versions. Treat this file as authoritative for the pinned version only.

---

## 1. Environment & auth

LangSmith reads configuration from environment variables (load via `pydantic-settings` / `.env`). The SDK `Client()` and all `@traceable` tracing pick these up automatically — you rarely pass keys explicitly. Internally the SDK resolves each variable through a `LANGSMITH_*` → `LANGCHAIN_*` namespace fallback (`langsmith.utils.get_env_var`), so the `LANGSMITH_`-prefixed name takes precedence and the legacy `LANGCHAIN_`-prefixed name is honored as a fallback.

| Variable | Purpose | Notes |
|---|---|---|
| `LANGSMITH_API_KEY` | API key for the SDK and tracing. | Required for any network call (`evaluate`, dataset/prompt/queue ops). Legacy alias `LANGCHAIN_API_KEY`. |
| `LANGSMITH_TRACING` | `"true"` enables background tracing of `@traceable` runs and `init_chat_model` calls. | Set `"false"` for pure-unit evaluator tests to avoid network. Legacy alias `LANGCHAIN_TRACING_V2` (note the `V2` suffix) still honored. |
| `LANGSMITH_PROJECT` | Default project (tracer session) runs are logged to. | `evaluate()` writes to an *experiment*, not this project; this governs ad-hoc traced runs and the online simulator. Legacy alias `LANGCHAIN_PROJECT`. |
| `LANGSMITH_ENDPOINT` | API base URL. | Default `https://api.smith.langchain.com`. Set for EU region (`https://eu.api.smith.langchain.com`) or self-hosted. Legacy alias `LANGCHAIN_ENDPOINT`. |

Provider keys for the apps-under-test and judges (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) are consumed by `langchain.init_chat_model`, not by LangSmith itself.

```python
from langsmith import Client
client = Client()  # reads LANGSMITH_API_KEY / LANGSMITH_ENDPOINT from env
```

---

## 2. Datasets & examples

### `Client.create_dataset`

```
create_dataset(self, dataset_name, *, description=None,
               data_type=DataType.kv, inputs_schema=None, outputs_schema=None,
               transformations=None, metadata=None) -> Dataset
```

Dataset names must be unique per workspace — `create_dataset` raises on a duplicate name. For idempotent setup, guard with `client.has_dataset(dataset_name=...)` (signature: `has_dataset(*, dataset_name=None, dataset_id=None) -> bool`) or read-or-create.

```python
ds = client.create_dataset("classify-intents", description="Intent classification eval set")
```

### `Client.create_examples` (bulk, preferred)

```
create_examples(self, *, dataset_name=None, dataset_id=None,
                examples=None, dangerously_allow_filesystem=False, **kwargs)
                -> UpsertExamplesResponse | dict
```

`examples` is a sequence of dicts (or `ExampleCreate`). **Confirmed example payload shape** (`schemas.ExampleCreate` fields): `id`, `created_at`, `inputs`, `outputs`, `metadata`, `split`, `attachments`, `use_source_run_io`, `use_source_run_attachments`, `source_run_id`. The three that matter for offline evals:

- `inputs: dict` — passed to your `target` function.
- `outputs: dict` — the **reference** outputs; surfaced to evaluators as `reference_outputs` (and as `example.outputs`).
- `split: str | list[str]` — tag(s) used for `list_examples(splits=[...])` and split-scoped experiments.

```python
client.create_examples(
    dataset_name="classify-intents",
    examples=[
        {"inputs": {"text": "cancel my plan"},
         "outputs": {"label": "cancellation"},
         "split": ["train"]},
        {"inputs": {"text": "where is my refund"},
         "outputs": {"label": "billing"},
         "split": ["test"]},
    ],
)
```

> Version note: in 0.8.16 the bulk signature is keyword-only (`examples=[...]`). The older positional `create_examples(inputs=[...], outputs=[...], dataset_id=...)` form from 0.1.x is **not** the shape here.

### `Client.create_example` (single)

```
create_example(self, inputs=None, dataset_id=None, dataset_name=None,
               created_at=None, outputs=None, metadata=None,
               split=None, example_id=None, source_run_id=None,
               use_source_run_io=False, use_source_run_attachments=None,
               attachments=None) -> Example
```

### `Client.list_examples`

```
list_examples(self, dataset_id=None, dataset_name=None, example_ids=None,
              as_of=None, splits=None, inline_s3_urls=True, *,
              offset=0, limit=None, metadata=None, filter=None,
              include_attachments=False, **kwargs) -> Iterator[Example]
```

- `splits=["test"]` — version-stable way to evaluate a single split.
- `as_of` — pin to a dataset **version** (datetime or version tag) for reproducible runs.

```python
test = list(client.list_examples(dataset_name="classify-intents", splits=["test"]))
```

---

## 3. Offline experiments — `evaluate()` / `aevaluate()`

### `langsmith.evaluate` (alias of `langsmith.evaluation.evaluate`)

```
evaluate(target, /, data=None, evaluators=None, summary_evaluators=None,
         metadata=None, experiment_prefix=None, description=None,
         max_concurrency=0, num_repetitions=1, client=None,
         blocking=True, experiment=None, upload_results=True, **kwargs)
         -> ExperimentResults | ComparativeExperimentResults
```

- **`target`** — a callable `inputs: dict -> outputs: dict`, a `Runnable`, or an existing experiment name/id (for re-eval). The type union also accepts a 2-tuple of experiments.
- **`data`** — dataset name, dataset id, `Dataset`, or an iterable of `Example`. Accepts the result of `list_examples(...)` for split-scoped runs.
- **`evaluators`** — row-level evaluators (Section 5).
- **`summary_evaluators`** — experiment-level aggregates: macro-F1, precision/recall (Section 6).
- **`num_repetitions`** (default `1`) — run each example *N* times; drives variance/noise reporting.
- **`max_concurrency`** (default `0`, typed `Optional[int]`) — `0` is the SDK default (no caller-set parallelism cap); pass a positive int to parallelize target+evaluator calls.
- **`experiment_prefix`** — human-readable experiment name prefix in the UI.
- **`metadata`** — dict attached to the experiment (e.g. `{"model": MODEL, "git_sha": ...}`); use it to filter/compare experiments.
- **`blocking`** (default `True`) — when `False`, returns immediately and uploads in the background.

```python
from langsmith import evaluate

results = evaluate(
    classify_app,                       # inputs:dict -> outputs:dict
    data="classify-intents",
    evaluators=[exact_match],
    summary_evaluators=[macro_f1, confusion_matrix],
    experiment_prefix="classify",
    metadata={"model": MODEL},
    num_repetitions=3,
    max_concurrency=8,
)
print(results.to_pandas())
```

### `langsmith.aevaluate` (async)

```
aevaluate(target, /, data=None, evaluators=None, summary_evaluators=None,
          metadata=None, experiment_prefix=None, description=None,
          max_concurrency=0, num_repetitions=1, client=None,
          blocking=True, experiment=None, upload_results=True, **kwargs)
          -> AsyncExperimentResults
```

Same parameters as `evaluate`; `target` may be an async callable / `AsyncIterable`. Use for async apps-under-test (e.g. the LangGraph agent).

```python
results = await aevaluate(async_agent, data="agent-tasks", evaluators=[trajectory_eval])
```

---

## 4. Pairwise / comparative — `evaluate_comparative()`

```
evaluate_comparative(experiments, /, evaluators,
                     experiment_prefix=None, description=None,
                     max_concurrency=5, client=None, metadata=None,
                     load_nested=False, randomize_order=False)
                     -> ComparativeExperimentResults
```

- **`experiments`** — a **2-tuple** of two prior experiment names/ids (e.g. two prompt-hub versions of the `generate` scenario).
- **`evaluators`** — comparative evaluators (Section 5.3) that rank/score the pair.
- **`randomize_order`** — shuffle which candidate is "A" vs "B" to debias position-sensitive judges.

```python
from langsmith.evaluation import evaluate_comparative

evaluate_comparative(
    ("generate-v1", "generate-v2"),
    evaluators=[pairwise_judge],
    randomize_order=True,
)
```

> **Version-critical:** in 0.8.16 there is **no `aevaluate_comparative`**. Comparative/pairwise evaluation is **synchronous only**. Note also that `evaluate_comparative`'s `max_concurrency` defaults to `5` (unlike `evaluate`'s `0`).

---

## 5. Evaluator contracts (row-level)

In 0.8.16 the evaluator-function arguments are **resolved by parameter name**. Confirmed supported argument names (pick any subset; order doesn't matter), from `supported_args` in `langsmith.evaluation.evaluator`:

```
run, example, inputs, outputs, reference_outputs, attachments
```

Mapping (confirmed from the SDK's argument binder):

| Param | Bound value |
|---|---|
| `run` | the full `Run` object (use for run-tree / trajectory inspection) |
| `example` | the full `Example` object |
| `inputs` | `example.inputs` |
| `outputs` | `run.outputs` (your target's return) |
| `reference_outputs` | `example.outputs` (the gold answer) |
| `attachments` | `example.attachments` |

The modern, recommended style uses `inputs / outputs / reference_outputs`. A function whose params are exactly `(run, example)` is also accepted for backward compatibility.

### 5.1 Return values

Confirmed coercion rules (`_format_evaluator_result`). An evaluator may return:

- a **`dict`** with keys from `EvaluationResult` — most importantly `key` (feedback name), `score` (numeric/bool), `value` (categorical/string), `comment` (rationale). Other accepted keys: `correction`, `evaluator_info`, `feedback_config`, `source_run_id`.
- a **bool / int / float** → coerced to `{"score": <x>}`.
- a **str** → coerced to `{"value": <x>}`.
- a **`list`** (of dicts) → emits **multiple feedback keys** in one evaluator (coerced to `{"results": [...]}`). This is exactly how the `extract` scenario emits `all_fields_present`, per-field match, and `field_accuracy` from a single evaluator.
- an **`EvaluationResult`** or **`EvaluationResults`** object (`EvaluationResults` is a `dict`/`TypedDict` with a single key `results: list[EvaluationResult]`).

An empty return (empty dict/list, or falsy) raises `ValueError`.

`EvaluationResult` fields (confirmed; it is a pydantic v1 `BaseModel`): `key` (**required**), `score`, `value`, `comment`, `correction`, `evaluator_info`, `feedback_config`, `source_run_id`, `target_run_id`, `extra`.

```python
# Single-key heuristic evaluator
def exact_match(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "exact_match",
            "score": int(outputs["label"] == reference_outputs["label"])}

# Multi-key code evaluator (extract scenario)
def field_eval(outputs: dict, reference_outputs: dict) -> list[dict]:
    pred, ref = outputs, reference_outputs
    results = [{"key": "all_fields_present", "score": int(isinstance(pred, dict))}]
    for f in ("merchant", "amount", "date"):
        results.append({"key": f"field:{f}",
                        "score": int(pred.get(f) == ref.get(f))})
    hits = sum(r["score"] for r in results[1:])
    results.append({"key": "field_accuracy", "score": hits / 3})
    return results
```

### 5.2 LLM-as-judge evaluators

Judges are ordinary row-level evaluators that call an `init_chat_model` judge with a structured Pydantic grade, then map it to feedback. Use `value` for categorical verdicts and `comment` for the rationale; attach `feedback_config` for continuous scores when you want UI range hints.

```python
def faithfulness_judge(inputs: dict, outputs: dict) -> dict:
    grade = judge_model.with_structured_output(FaithfulnessGrade).invoke(
        build_prompt(question=inputs["question"],
                     context=outputs["context"], answer=outputs["answer"]))
    return {"key": "faithfulness", "score": grade.score, "comment": grade.reasoning}
```

### 5.3 Comparative evaluator contract

Confirmed supported args for comparative evaluators (`supported_args` in `_normalize_comparative_evaluator_func`): `runs` (a sequence of the paired runs), plus `example`, `inputs`, `outputs`, `reference_outputs` (note: singular `example`, unlike the plural `examples` of summary evaluators). Return a dict with a `key` and a `scores` mapping of run IDs to numeric scores (or a `ComparisonEvaluationResult` object).

```python
def pairwise_judge(runs, inputs) -> dict:
    a, b = runs[0].outputs, runs[1].outputs
    winner = judge_model.with_structured_output(Pref).invoke(prompt(inputs, a, b))
    return {"key": "preference",
            "scores": {runs[0].id: int(winner == "A"),
                       runs[1].id: int(winner == "B")}}
```

---

## 6. Summary evaluators (experiment-level)

Confirmed supported argument names (`supported_args` for the summary-evaluator normalizer):

```
runs, examples, inputs, outputs, reference_outputs
```

These receive **sequences** (one element per evaluated example): `outputs` is `[run.outputs, ...]`, `reference_outputs` is `[example.outputs, ...]`. Return a single `dict` / `EvaluationResult` (or `EvaluationResults`) — same coercion rules as row-level. This is where macro-F1, precision/recall, and accuracy live.

```python
def macro_f1(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    y_pred = [o["label"] for o in outputs]
    y_true = [r["label"] for r in reference_outputs]
    return {"key": "macro_f1", "score": _macro_f1(y_true, y_pred)}
```

The `(runs, examples)` two-arg form is also accepted for backward compatibility.

---

## 7. Prompt Hub — pull / push

### `Client.pull_prompt`

```
pull_prompt(self, prompt_identifier, *, include_model=False) -> Any
```

`prompt_identifier` supports version pinning: `"name"`, `"name:<commit-hash>"`, or `"name:<tag>"`. Returns a runnable prompt (a `ChatPromptTemplate`; or a prompt-bound model when `include_model=True`). Used by the `generate` scenario to fetch versioned prompts.

```python
prompt = client.pull_prompt("joao/summarize:prod")
chain = prompt | get_chat_model()
```

### `Client.push_prompt`

```
push_prompt(self, prompt_identifier, *, object=None, parent_commit_hash="latest",
            is_public=None, description=None, readme=None, tags=None) -> str
```

Pushes (and versions) a prompt; returns the URL of the new commit. `object` is the `ChatPromptTemplate` to store.

```python
url = client.push_prompt("joao/summarize", object=my_prompt, tags=["v2"])
```

---

## 8. Annotation queues (human grading)

Confirmed two-step SDK flow.

### `Client.create_annotation_queue`

```
create_annotation_queue(self, *, name, description=None,
                        queue_id=None, rubric_instructions=None)
                        -> AnnotationQueueWithDetails
```

### `Client.add_runs_to_annotation_queue`

```
add_runs_to_annotation_queue(self, queue_id, *, run_ids) -> None
```

`run_ids` is a `list` of run UUIDs. Combine with `client.list_runs(...)` + sampling to enqueue a representative subset.

```python
q = client.create_annotation_queue(name="classify-human-review",
                                   rubric_instructions="Grade label correctness 0-1.")
runs = list(client.list_runs(project_name="classify-online", limit=200))
sample = random.sample(runs, 20)
client.add_runs_to_annotation_queue(q.id, run_ids=[r.id for r in sample])
```

Grading itself (assigning reviewers, applying the rubric) happens in the **LangSmith UI**.

---

## 9. pytest testing integration (CI regression gate)

Confirmed in 0.8.16:

- There is **no** importable `langsmith.pytest` or `pytest_langsmith` module. The plugin is **auto-registered** via the `pytest11` entry point `langsmith_plugin -> langsmith.pytest_plugin` (verified discoverable; `langsmith.pytest_plugin` is importable; no explicit import or `-p` flag needed).
- Mark tests with `@pytest.mark.langsmith`.
- Log via `from langsmith import testing as t`. Confirmed exports & signatures:

```
t.log_inputs(inputs: dict, /) -> None
t.log_outputs(outputs: dict, /) -> None
t.log_reference_outputs(reference_outputs: dict, /) -> None
t.log_feedback(feedback: dict | list[dict] | None = None, /, *, key: str,
               score=None, value=None, **kwargs) -> None
t.trace_feedback(*, name: str = "Feedback")  # context manager
```

```python
import pytest
from langsmith import testing as t

@pytest.mark.langsmith
def test_classify_regression():
    inputs = {"text": "cancel my plan"}
    t.log_inputs(inputs)
    t.log_reference_outputs({"label": "cancellation"})

    out = classify_app(inputs)
    t.log_outputs(out)

    score = int(out["label"] == "cancellation")
    t.log_feedback(key="exact_match", score=score)
    assert score == 1  # the CI gate: a hard pass/fail boundary
```

Run with `pytest`; results stream to LangSmith as a test experiment while standard pass/fail still gates the GitHub Actions job. Pure-unit evaluator tests (no `@pytest.mark.langsmith`, `LANGSMITH_TRACING=false`) run fully offline.

> **What this repo actually does.** The above is the *native* integration. This repo's
> gates instead drive a full `evaluate()` run and assert on the aggregate feedback, marked
> with a **custom `@pytest.mark.live`** marker — the name `langsmith` is reserved by the
> auto-registered plugin, and `live` cleanly separates credential-needing gates (`pytest -m live`)
> from the default offline unit tests. Both approaches are valid; the `live` route keeps the
> dataset-level threshold logic explicit in `tests/_gate_helpers.py`.

---

## 10. Online evals (production-traffic simulator) — SDK vs UI

| Concern | Mechanism | SDK or UI |
|---|---|---|
| Emit traced production-style runs to a project | `@traceable` (importable directly from `langsmith` top level — it is in `langsmith.__all__`) wrapping the app; runs land in `LANGSMITH_PROJECT`. | **SDK** |
| Attach feedback to a live run programmatically | `client.create_feedback(run_id, key=..., score=..., value=..., comment=...)` | **SDK** |
| Define an **online evaluator / automation rule** (sample N% of incoming runs, auto-run an LLM-judge) | Rule creation, sampling rate, and evaluator binding are configured on the project's **Rules / Automations** tab. | **UI** (document the click-path in `docs/`) |

```python
from langsmith import traceable  # also available as: from langsmith.run_helpers import traceable

@traceable(run_type="chain", name="classify")
def classify_app(inputs: dict) -> dict:
    ...  # traced automatically when LANGSMITH_TRACING=true
```

> Version note: `traceable` and `trace` are defined in `langsmith.run_helpers` and **re-exported at the `langsmith` top level** (both appear in `langsmith.__all__` in 0.8.16), so `from langsmith import traceable, trace` is the canonical import; `from langsmith.run_helpers import traceable` also works. The online **automation rule** is the one piece of the showcase that cannot be fully scripted — create it in the UI and capture screenshots + a deep-link in the README.

---

### Quick confirmation log (introspected on `langsmith==0.8.16`)

- `langsmith.evaluate is langsmith.evaluation.evaluate` → `True`; `langsmith.aevaluate` exists.
- `langsmith.evaluation` exports `evaluate`, `aevaluate`, `evaluate_comparative` — **no** `aevaluate_comparative`.
- `traceable` and `trace` are in `langsmith.__all__` (re-exported from `langsmith.run_helpers`).
- Row-level evaluator args: `run, example, inputs, outputs, reference_outputs, attachments`.
- Comparative evaluator args: `runs, example, inputs, outputs, reference_outputs` (singular `example`); returns `{key, scores: {run_id: score}}` or `ComparisonEvaluationResult`.
- Summary evaluator args: `runs, examples, inputs, outputs, reference_outputs`.
- Result coercion: `bool|int|float -> {"score": x}`; `str -> {"value": x}`; `list[dict] -> {"results": [...]}`; `dict` passthrough; empty/falsy raises.
- `EvaluationResult` (pydantic v1 `BaseModel`): `key` is the only required field; full fields `key, score, value, comment, correction, evaluator_info, feedback_config, source_run_id, target_run_id, extra`. `EvaluationResults` is a `dict`/`TypedDict` `{results: list[EvaluationResult]}`.
- `evaluate.max_concurrency` default `0`; `evaluate_comparative.max_concurrency` default `5`.
- Client has `create_dataset`, `has_dataset`, `create_examples`, `create_example`, `list_examples`, `pull_prompt`, `push_prompt`, `create_annotation_queue`, `add_runs_to_annotation_queue`, `create_feedback`, `list_runs`.
- `schemas.ExampleCreate` fields: `id, created_at, inputs, outputs, metadata, split, attachments, use_source_run_io, use_source_run_attachments, source_run_id`.
- pytest plugin entry point `langsmith_plugin -> langsmith.pytest_plugin` is discoverable; `langsmith.pytest` / `pytest_langsmith` are **not** importable. `langsmith.testing` exports `log_inputs`, `log_outputs`, `log_reference_outputs`, `log_feedback`, `trace_feedback`.
