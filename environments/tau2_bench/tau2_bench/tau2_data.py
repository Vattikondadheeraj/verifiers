"""Dataset builder: convert tau2 tasks into verifiers-compatible HuggingFace Datasets."""

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
) -> Dataset:
    """Build dataset for user agent training.

    Each row contains the user's preference context as the prompt.
    The user agent learns to negotiate with a fixed platform agent.
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
        prompt_text = _build_user_prompt(task, user_data)
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


def _build_user_prompt(task: Task, user_data: dict) -> str:
    """Build the prompt for the user agent given preference context."""
    parts = ["You are a personal travel assistant booking a flight for your user."]

    emails = user_data.get("emails", [])
    if emails:
        email_text = "\n\n".join(
            f"Subject: {e.get('subject', '')}\n{e.get('body', '')}" for e in emails
        )
        parts.append(f"\nUser's email history:\n{email_text}")

    profile = user_data.get("preference_profile")
    if profile:
        parts.append(f"\nPreference profile:\n{json.dumps(profile, indent=2)}")

    reason = user_data.get("reason_for_call")
    if reason:
        parts.append(f"\nReason for call: {reason}")

    parts.append(
        "\nCall the airline customer service and book a flight "
        "matching these preferences. Be clear about what you want."
    )
    return "\n".join(parts)


def _transpose(rows: list[dict]) -> dict[str, list]:
    """Convert list-of-dicts to dict-of-lists for Dataset.from_dict."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}
