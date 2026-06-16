# langsmith-evals-showcase

A multi-scenario **eval gym** that exercises the full surface of [LangSmith](https://docs.smith.langchain.com/) evaluation in one coherent, runnable repository. Five small, independent apps-under-test each spotlight a different family of eval capability — offline `evaluate()` with heuristic and summary evaluators, custom code evaluators emitting many feedback keys, retrieval metrics, custom LLM-as-judge graders, agent-trajectory evaluation, Prompt Hub versioning, and pairwise comparison — wrapped by cross-cutting online evals, an SDK-built annotation queue, and a pytest CI regression gate. Everything is provider-agnostic (swap the model with one env var) and the display layer is LangSmith itself: no custom dashboard, just well-crafted experiments you can open and read.

---

## What this demonstrates

Rows are LangSmith capabilities; columns are the five apps-under-test plus the cross-cutting concerns. A cell marks where each capability is exercised.

| LangSmith capability | classify | extract | rag | agent | generate | cross-cutting |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Datasets, splits, versions (`as_of`) | ● | ● | ● | ● | ● | |
| Offline `evaluate()` / `aevaluate()` | ● | ● | ● | ● | ● | |
| Heuristic exact-match evaluator | ● | | | | | |
| Multi-key **code** evaluator (one fn, many feedback keys) | | ● | | | | |
| Numeric-tolerance / normalized field match | | ● | | | | |
| **Summary** evaluators (macro-F1, precision/recall, confusion matrix) | ● | | | | | |
| `num_repetitions` variance / noise reporting | ● | | | | | |
| Retrieval recall@k (heuristic over intermediate context) | | | ● | | | |
| **Custom LLM-as-judge** (structured Pydantic grade) | | | ● | ● | ● | |
| Reference-based answer-correctness judge | | | ● | | | |
| Faithfulness / groundedness / context-relevance | | | ● | | | |
| **Trajectory** evaluation (tool sequence, run-tree inspection) | | | | ● | | |
| Tool-selection correctness + final-answer judge | | | | ● | | |
| **Prompt Hub** pull with version pinning | | | | | ● | |
| Reference-free multi-criteria rubric judge | | | | | ● | |
| **Pairwise** `evaluate_comparative()` across versions | | | | | ● | |
| **Online** evals (traffic simulator + automation rule) | | | | | | ● |
| **Annotation queue** built via SDK, sampled runs enqueued | | | | | | ● |
| pytest **CI regression gate** (`live` marker) | ● | | | | | ● |

---

## Architecture

**Provider-agnostic.** Both the apps-under-test and the LLM-judge evaluators are built on `langchain.init_chat_model`, configured with a string like `"anthropic:claude-haiku-4-5"` or `"openai:gpt-4.1-mini"`. Two env vars drive everything: `APP_MODEL` (the system under test) and `JUDGE_MODEL` (the grader). Switch providers without touching code. Traces flow to LangSmith natively via `LANGSMITH_TRACING`.

**Eval-gym layout.** Each scenario is a self-contained package under `src/evals_showcase/scenarios/<name>/` with an `app.py` (the thing being evaluated), a `data.jsonl` (the dataset), and an `experiment.py` (dataset seeding + the `evaluate()` wiring). Shared machinery — config, model factories, embeddings, dataset helpers, and the evaluator library — lives one level up and is reused across scenarios. New scenarios register themselves in `cli.py` (`REGISTRY`) and immediately become available to `evals seed` / `evals run`.

**Display is the LangSmith UI.** This repo deliberately builds **no** custom dashboard or report generator. Experiments, feedback charts, comparison views, traces, and the confusion matrix all render in LangSmith; the README narrates results with screenshots and deep-links. The code stays focused on eval craft.

---

## The five scenarios

### 1. `classify` — intent / category classifier
A single structured-output LLM call that labels an utterance. The reference scenario for **dataset mechanics and aggregate metrics**: train/test splits, a row-level heuristic exact-match evaluator, and **summary evaluators** computing macro-F1, per-class precision/recall, and a confusion matrix. Runs with `num_repetitions` to surface variance, and is gated in CI via the `live` pytest marker.

### 2. `extract` — text → JSON field extraction
Parses transaction-style text into a Pydantic model. Spotlights a **custom code evaluator that emits multiple feedback keys from one function** — `json-validity`, per-field normalized match, numeric tolerance on amounts, and an aggregate `field-accuracy` — returned as a `list[dict]` so each surfaces as its own feedback metric in the UI.

### 3. `rag` — tiny retriever + answer generation
A small in-memory retriever (local `fastembed` embeddings, cosine over a committed corpus) feeding answer generation. Combines a **heuristic retrieval recall@k** over the intermediate retrieved context with **custom LLM-as-judge** evaluators for faithfulness/groundedness, reference-based answer-correctness, and context-relevance — demonstrating evaluation of both intermediate and final outputs.

### 4. `agent` — LangGraph ReAct agent
A small ReAct agent with 2–3 tools. Spotlights **trajectory evaluation**: inspecting the run tree to score the tool-call sequence, checking tool-selection correctness, and judging the final answer. Uses `aevaluate()` for the async target.

### 5. `generate` — summarize / draft from Prompt Hub
A drafting task whose prompt is pulled from the **LangSmith Prompt Hub** with version pinning. Spotlights a reference-free, multi-criteria **rubric judge** (categorical + continuous feedback) and **pairwise comparison** via `evaluate_comparative()` across two prompt versions, viewed in the comparison UI.

### Cross-cutting
- **Online evals** — a production-traffic simulator emits `@traceable` runs into a project; an automation rule samples incoming traffic and auto-runs a judge. Rule creation is UI-driven and documented (with click-path + screenshots) in `docs/`.
- **Annotation queue** — created via the SDK, with a sampled subset of runs enqueued for human grading against a rubric.
- **CI regression gate** — a pytest job (the `live` marker) that streams a test experiment to LangSmith while standard pass/fail gates the GitHub Actions run.

---

## Quickstart

Prereqs: [uv](https://docs.astral.sh/uv/) and a LangSmith account.

```bash
# 1. Install (creates the venv, installs deps incl. dev)
uv sync

# 2. Configure credentials
cp .env.example .env
#   then edit .env and set, at minimum:
#     LANGSMITH_API_KEY=lsv2_...
#     ANTHROPIC_API_KEY=sk-ant-...           # or OPENAI_API_KEY=sk-...
#     APP_MODEL=anthropic:claude-haiku-4-5    # the system under test
#     JUDGE_MODEL=anthropic:claude-sonnet-4-6 # the LLM-as-judge

# 3. Push datasets to LangSmith (idempotent)
make seed

# 4. Run a scenario's offline experiment
make eval SCENARIO=classify
```

Then open the experiment in LangSmith from the link printed to the console.

Other useful targets (`make help` lists them all):

```bash
make test       # deterministic unit tests (runs `pytest -m 'not live'`, no API calls)
make gate       # LangSmith-backed regression gates (pytest -m live; requires credentials)
make lint       # ruff check
make format     # ruff format + ruff check --fix
make typecheck  # mypy
```

The CLI is also callable directly: `uv run evals list`, `uv run evals seed <scenario|all>` (defaults to `all`), `uv run evals run <scenario> -r 3`.

---

## Project layout

```
langsmith-evals-showcase/
├── src/evals_showcase/
│   ├── config.py            # pydantic-settings: env → typed config
│   ├── models.py            # get_chat_model / get_judge_model (init_chat_model)
│   ├── embeddings.py        # fastembed local embeddings (rag)
│   ├── datasets.py          # idempotent dataset create / version / split
│   ├── evaluators/
│   │   ├── heuristic.py     # exact-match, field match, recall@k
│   │   ├── judges.py        # custom LLM-as-judge (structured Pydantic grades)
│   │   ├── summary.py       # macro-F1, precision/recall, confusion matrix
│   │   └── trajectory.py    # agent tool-sequence / run-tree evaluators
│   ├── scenarios/
│   │   ├── classify/        # app.py · data.jsonl · experiment.py
│   │   ├── extract/         #   "
│   │   ├── rag/             #   "
│   │   ├── agent/           #   "
│   │   └── generate/        #   "
│   ├── online.py            # production-traffic simulator (@traceable)
│   ├── annotation.py        # annotation queue created + populated via SDK
│   └── cli.py               # typer: evals list / seed / run
├── tests/                   # `live`-marked gates + pure-unit evaluator tests
├── docs/
│   ├── EVALUATION_STRATEGY.md
│   ├── LANGSMITH_API_REFERENCE.md
│   └── images/              # screenshots
├── .github/                 # GitHub Actions CI regression gate
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Documentation

- **[docs/EVALUATION_STRATEGY.md](docs/EVALUATION_STRATEGY.md) — the centerpiece.** The opinionated walkthrough of *why* each scenario evaluates what it does: when to reach for heuristic vs. summary vs. LLM-judge evaluators, how to think about variance and repetitions, judge-design pitfalls, pairwise vs. absolute grading, and where `openevals` would slot in as an alternative (it is referenced, **not** a dependency — judges here are custom). Start here.
- **[docs/LANGSMITH_API_REFERENCE.md](docs/LANGSMITH_API_REFERENCE.md)** — a concise, version-grounded cheat-sheet of the exact LangSmith APIs this repo uses, every signature confirmed by introspection against the pinned `langsmith==0.3.45`.

---

## Results

Experiments live in **LangSmith** — that is the display layer, by design. Each scenario writes a named experiment you can open to inspect per-row feedback, summary metrics, traces, and comparison views. The screenshots and deep-links below will be added as scenarios land.

> **TODO — screenshots pending.** The images below are placeholders; the files do not exist yet. Replace with captures from real experiment runs and link each to its LangSmith experiment.

| View | Screenshot |
|---|---|
| `classify` — confusion matrix + macro-F1 summary | _TODO: `docs/images/classify-summary.png`_ |
| `extract` — per-field feedback keys on a single experiment | _TODO: `docs/images/extract-feedback.png`_ |
| `rag` — LLM-judge faithfulness + recall@k | _TODO: `docs/images/rag-judges.png`_ |
| `agent` — trajectory / run-tree evaluation | _TODO: `docs/images/agent-trajectory.png`_ |
| `generate` — pairwise comparison view | _TODO: `docs/images/generate-pairwise.png`_ |
| online evals — automation rule + sampled feedback | _TODO: `docs/images/online-evals.png`_ |

---

## Build status & roadmap

Scenarios ship **incrementally**; the shared eval machinery and the first scenario are wired end-to-end, with the rest landing in sequence. Each ✅ is fully runnable (`evals seed` + `evals run`) with its evaluators and tests in place.

| Scenario / concern | Status |
|---|---|
| Shared core (config, models, embeddings, datasets, evaluator library) | ✅ |
| `classify` — splits, summary metrics, repetitions, CI gate | ✅ |
| `extract` — multi-key code evaluator | ✅ |
| `rag` — retrieval metrics + LLM-judges | ✅ |
| `agent` — trajectory evaluation | ✅ |
| `generate` — Prompt Hub + pairwise comparison | ✅ |
| Online evals (simulator + automation rule) | ⏳ planned |
| Annotation queue (SDK) | ⏳ planned |
| GitHub Actions regression gate | ✅ |

---

Author: **João Pedro Gomes Ferreira** ([@Jpgomesf](https://github.com/Jpgomesf)) · License: MIT
