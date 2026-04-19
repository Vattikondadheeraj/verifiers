"""Dataset builder: convert tau2 tasks into verifiers-compatible HuggingFace Datasets.

Supports three modes:
  - platform: train the customer service agent (single-airline)
  - user: train the user agent (single-airline, text-based)
  - marketplace: train the user agent (multi-airline, tool-based)
"""

from __future__ import annotations

import json
from typing import Optional

from datasets import Dataset

from tau2.data_model.tasks import Task
from tau2.domains.airline_a2a.environment import get_tasks
from tau2.domains.airline_a2a.utils import load_user_data


def build_platform_dataset(
    num_tasks: int = -1,
    task_split_name: Optional[str] = "base",
    email_prefs_path: Optional[str] = None,
    preference_profiles_path: Optional[str] = None,
) -> Dataset:
    """Build dataset for platform agent training.

    Each row contains the user's opening message as the prompt and task
    metadata in ``info`` for rollout-time state setup and evaluation.
    """
    tasks = _load_tasks(num_tasks, task_split_name)
    rows = []
    for task in tasks:
        user_msg = _get_initial_user_message(task)
        user_data = load_user_data(
            task,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
        )
        rows.append(
            {
                "prompt": json.dumps([{"role": "user", "content": user_msg}]),
                "answer": task.id,
                "task": "platform",
                "info": json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "user_data": user_data,
                    }
                ),
            }
        )
    return Dataset.from_dict(_transpose(rows))


def build_user_dataset(
    num_tasks: int = -1,
    task_split_name: Optional[str] = "base",
    email_prefs_path: Optional[str] = None,
    preference_profiles_path: Optional[str] = None,
    disclosure_level: str = "natural",
) -> Dataset:
    """Build dataset for user agent training (single-airline).

    Each row contains the user's preference context as the prompt.
    The user agent learns to negotiate with a fixed platform agent.

    Args:
        disclosure_level: How much the user reveals in the first message.
            "full" — all preferences as bullet points
            "partial" — route + dates + user_id only
            "minimal" — just "Book me a flight" + user_id
            "natural" — top N preferences volunteered, rest held back
    """
    tasks = _load_tasks(num_tasks, task_split_name)
    rows = []
    for task in tasks:
        user_data = load_user_data(
            task,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
        )
        if user_data is None:
            user_data = {}
        prompt_text = _build_user_prompt(task, user_data, disclosure_level=disclosure_level)
        rows.append(
            {
                "prompt": json.dumps([{"role": "user", "content": prompt_text}]),
                "answer": task.id,
                "task": "user",
                "info": json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "user_data": user_data,
                    }
                ),
            }
        )
    if not rows:
        return Dataset.from_dict({"prompt": [], "answer": [], "task": [], "info": []})
    return Dataset.from_dict(_transpose(rows))


def build_marketplace_dataset(
    num_tasks: int = -1,
    task_split_name: Optional[str] = "base",
    email_prefs_path: Optional[str] = None,
    preference_profiles_path: Optional[str] = None,
    disclosure_level: str = "natural",
) -> Dataset:
    """Build dataset for marketplace user training (multi-airline).

    The user agent gets marketplace tools (list_airlines, query_airline, get_user_details)
    and shops across multiple airlines. The prompt includes email-based preferences
    and a marketplace greeting.
    """
    tasks = _load_tasks(num_tasks, task_split_name)
    rows = []
    for task in tasks:
        # Only include tasks that have sampled_policies (marketplace-ready)
        gen_meta = task.generation_metadata or {}
        if not gen_meta.get("sampled_policies"):
            continue

        user_data = load_user_data(
            task,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
        )
        if user_data is None:
            user_data = {}

        # Marketplace prompt: user gets a greeting from the marketplace system
        prompt_text = _build_marketplace_prompt(task, user_data, disclosure_level=disclosure_level)

        rows.append(
            {
                "prompt": json.dumps([
                    {"role": "user", "content": prompt_text},
                ]),
                "answer": task.id,
                "task": "marketplace",
                "info": json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "user_data": user_data,
                    }
                ),
            }
        )
    if not rows:
        return Dataset.from_dict({"prompt": [], "answer": [], "task": [], "info": []})
    return Dataset.from_dict(_transpose(rows))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tasks(num_tasks: int, task_split_name: Optional[str]) -> list[Task]:
    tasks = get_tasks(task_split_name=task_split_name)
    if num_tasks > 0:
        tasks = tasks[:num_tasks]
    return tasks


def _get_initial_user_message(task: Task) -> str:
    """Extract the opening user message from a task."""
    if task.initial_state and task.initial_state.message_history:
        for msg in task.initial_state.message_history:
            if msg.role == "user" and msg.content:
                return msg.content

    instructions = task.user_scenario.instructions
    if isinstance(instructions, str):
        return instructions
    return (
        f"{instructions.reason_for_call}\n\n"
        f"Known info:\n{instructions.known_info or 'N/A'}"
    )


def _build_user_prompt(task: Task, user_data: dict, disclosure_level: str = "natural") -> str:
    """Build the prompt for the user agent given preference context."""
    parts = ["You are a personal travel assistant booking a flight for your user."]

    emails = user_data.get("emails", [])
    if emails:
        email_text = "\n\n".join(
            f"Subject: {e.get('subject', '')}\n{e.get('body', '')}" for e in emails
        )
        parts.append(f"\nUser's email history:\n{email_text}")

    reason = user_data.get("reason_for_call")
    if reason:
        parts.append(f"\nReason for call: {reason}")

    # Disclosure level controls how much the user reveals upfront
    if disclosure_level == "full":
        profile = user_data.get("preference_profile")
        if profile:
            prefs = profile.get("preferences", {})
            pref_lines = [f"- {k}: {v}" for k, v in prefs.items()]
            parts.append(f"\nUser preferences (share all of these with the agent):\n" + "\n".join(pref_lines))
    elif disclosure_level == "partial":
        parts.append(
            "\nShare only the route, dates, and your user ID upfront. "
            "Wait for the agent to ask about other preferences."
        )
    elif disclosure_level == "minimal":
        parts.append(
            "\nJust say you need to book a flight and provide your user ID. "
            "Let the agent guide the conversation."
        )
    else:  # natural (default)
        parts.append(
            "\nShare your most important preferences naturally in conversation. "
            "You don't need to list everything upfront — respond to the agent's questions."
        )

    parts.append(
        "\nCall the airline customer service and book a flight "
        "matching these preferences. Be clear about what you want."
    )
    return "\n".join(parts)


def _build_marketplace_prompt(task: Task, user_data: dict, disclosure_level: str = "natural") -> str:
    """Build the prompt for marketplace user training."""
    parts = [
        "Welcome to the airline marketplace! You can browse multiple airlines "
        "to find the best flight deal. Use your tools to get started.\n\n"
        "REMINDER: Call list_airlines() now to see available airlines, "
        "then use query_airline(airline_id, message) to talk to them."
    ]

    # The actual preferences and emails go into the system prompt via _build_user_context
    # in the env's setup_state. The prompt here is just the marketplace greeting.
    return "\n".join(parts)


def _transpose(rows: list[dict]) -> dict[str, list]:
    """Convert list-of-dicts to dict-of-lists for Dataset.from_dict."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}
