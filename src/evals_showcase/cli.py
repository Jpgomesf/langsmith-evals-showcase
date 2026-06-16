"""Command-line entrypoint: ``evals seed`` and ``evals run <scenario>``.

New scenarios become available by adding them to ``REGISTRY``; each registered
module must expose ``seed(client)`` and ``run_experiment(**kwargs)``.
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from .config import get_settings
from .scenarios.agent import experiment as agent
from .scenarios.classify import experiment as classify
from .scenarios.extract import experiment as extract
from .scenarios.generate import experiment as generate
from .scenarios.rag import experiment as rag

app = typer.Typer(add_completion=False, help="Run the LangSmith eval-gym scenarios.")

# scenario name -> module exposing seed(client) / run_experiment(**kwargs)
REGISTRY = {
    "classify": classify,
    "extract": extract,
    "rag": rag,
    "agent": agent,
    "generate": generate,
}


@app.callback()
def _bootstrap() -> None:
    """Load ``.env`` so LangSmith and provider credentials are on the environment."""
    load_dotenv()


def _resolve(scenario: str) -> dict[str, object]:
    if scenario == "all":
        return dict(REGISTRY)
    if scenario not in REGISTRY:
        available = ", ".join(sorted(REGISTRY)) or "(none yet)"
        raise typer.BadParameter(f"unknown scenario '{scenario}'. available: {available}")
    return {scenario: REGISTRY[scenario]}


@app.command(name="list")
def list_scenarios() -> None:
    """List the scenarios that are wired up."""
    for name in sorted(REGISTRY):
        typer.echo(name)


@app.command()
def seed(scenario: str = typer.Argument("all", help="Scenario name or 'all'.")) -> None:
    """Push scenario dataset(s) to LangSmith (idempotent)."""
    from langsmith import Client

    client = Client()
    for name, module in _resolve(scenario).items():
        dataset_id = module.seed(client)  # type: ignore[attr-defined]
        typer.echo(f"seeded {name}: dataset {dataset_id}")


@app.command()
def run(
    scenario: str = typer.Argument(..., help="Scenario name to evaluate."),
    repetitions: int = typer.Option(1, "--repetitions", "-r", help="Runs per example."),
) -> None:
    """Run a scenario's offline evaluation experiment."""
    module = _resolve(scenario)[scenario]
    module.run_experiment(repetitions=repetitions)  # type: ignore[attr-defined]
    typer.echo(f"experiment submitted for '{scenario}' — view results in LangSmith.")


@app.command()
def online(
    n: int = typer.Option(20, "-n", help="Number of traced runs to send."),
    project: str = typer.Option("", help="Target project (default: <project>-online)."),
) -> None:
    """Send simulated production traffic to a project for online evaluation."""
    from .online import simulate

    count = simulate(n=n, project=project or None)
    typer.echo(f"sent {count} traced runs — attach an online evaluator rule in LangSmith.")


@app.command()
def annotate(
    project: str = typer.Option("", help="Project to sample runs from (default: configured)."),
    queue: str = typer.Option("evals-showcase-review", help="Annotation queue name."),
    limit: int = typer.Option(10, help="How many recent runs to enqueue."),
) -> None:
    """Create an annotation queue and enqueue recent runs for human review."""
    from langsmith import Client

    from .annotation import enqueue_from_project

    target = project or get_settings().langsmith_project
    created = enqueue_from_project(Client(), project=target, queue_name=queue, limit=limit)
    typer.echo(
        f"created queue '{queue}' ({created.id}); enqueued up to {limit} runs from '{target}'."
    )


if __name__ == "__main__":
    app()
