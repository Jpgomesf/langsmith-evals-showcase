# Online & Human Evaluation

The five scenarios run *offline* experiments against curated datasets. This file
covers the two cross-cutting pieces that round out a real eval stack: **online
evals** on production traffic and **human review** via annotation queues.

## Offline vs. online

| | Offline experiment | Online evaluation |
|---|---|---|
| Input | A fixed, curated **dataset** | Live **production traffic** (no ground truth) |
| Trigger | You run `evals run <scenario>` | A **rule** auto-runs on sampled incoming runs |
| Has references? | Usually yes | No — use reference-free evaluators |
| Question it answers | "Did this change regress quality?" | "How is the deployed system doing right now?" |

## Online evals

The repo can't create automation rules from the SDK (rule creation is UI-driven
in this LangSmith version), but it produces the traffic the rule scores.

1. **Generate traffic.** Send simulated, unlabeled production runs to a dedicated
   project:
   ```bash
   uv run evals online -n 30          # → project "<LANGSMITH_PROJECT>-online"
   ```
2. **Attach an online evaluator (LangSmith UI).**
   - Open the `…-online` project → **Rules** (a.k.a. Automations) → **+ New rule**.
   - Set a **sampling rate** (e.g. 100% for the demo, lower in production).
   - Add an **online evaluator** — an LLM-as-judge prompt scoring each run
     reference-free (e.g. "is this support reply helpful and on-topic?").
   - Save. New runs entering the project are scored automatically; the feedback
     shows up on each trace and in the project's charts.

Because production traffic has no labels, online evaluators must be
**reference-free** — exactly the kind built for `rag` (faithfulness) and
`generate` (rubric).

## Human review with annotation queues

Automated judges should be **calibrated against human labels** (see
`EVALUATION_STRATEGY.md`). Annotation queues collect a sample of runs for people
to grade against a rubric.

1. **Create a queue and enqueue runs:**
   ```bash
   uv run evals annotate --project "<project>" --queue evals-showcase-review --limit 10
   ```
   This creates a queue with a 1–5 quality rubric and adds the most recent runs.
2. **Review (LangSmith UI).** Open **Annotation Queues → evals-showcase-review**.
   Reviewers see one run at a time, score it against the rubric, and move on. The
   scores attach to the runs as feedback.
3. **Calibrate.** Compare human scores to the automated judge's scores on the same
   runs; if they disagree, refine the judge prompt (or the rubric). This closes the
   loop between cheap automated evals and trustworthy human judgment.
