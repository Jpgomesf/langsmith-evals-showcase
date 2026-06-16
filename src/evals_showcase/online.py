"""Online (production) evaluation — traffic simulator.

Sends traced, *unlabeled* runs to a dedicated project to mimic production
traffic. An online evaluator (automation rule) is then attached to that project
in the LangSmith UI to score incoming runs automatically — see
``docs/ONLINE_AND_HUMAN_EVAL.md`` for the click-path. Rule creation is UI-driven;
this module produces the traffic the rule runs against.
"""

from __future__ import annotations

from itertools import cycle, islice

from langsmith import traceable

from .config import get_settings
from .scenarios.classify.app import classify

# Realistic, label-free messages — production traffic has no ground truth.
SAMPLE_TRAFFIC = [
    "My card was declined but I was still charged twice.",
    "I can't log in even after resetting my password.",
    "Could you add a way to bulk-export my data?",
    "The dashboard has been loading forever this morning.",
    "Your phone support kept me on hold for an hour, awful.",
    "Hi! Just wanted to thank the team for the fast fix.",
    "How do I update the credit card on my account?",
    "The mobile app crashes whenever I open settings.",
]


def take_traffic(n: int) -> list[str]:
    """Return ``n`` sample messages, cycling through the pool as needed."""
    return list(islice(cycle(SAMPLE_TRAFFIC), n))


@traceable(name="classify-online", run_type="chain")
def _handle(text: str) -> str:
    """A traced 'production' request handler."""
    return classify(text)


def simulate(n: int = 20, project: str | None = None) -> int:
    """Send ``n`` traced runs to ``project`` (defaults to ``<project>-online``)."""
    target = project or f"{get_settings().langsmith_project}-online"
    for text in take_traffic(n):
        _handle(text, langsmith_extra={"project_name": target})
    return n
