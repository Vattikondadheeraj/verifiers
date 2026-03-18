"""Reward system wrapping tau2's evaluation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from copy import deepcopy
from typing import Any

import verifiers as vf
from verifiers.types import State

from tau2.data_model.message import (
    AssistantMessage as Tau2AssistantMsg,
    ToolCall as Tau2ToolCall,
    ToolMessage as Tau2ToolMsg,
    UserMessage as Tau2UserMsg,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory conversion
# ---------------------------------------------------------------------------


def _vf_messages_to_tau2(messages: list[dict[str, Any]]) -> list:
    """Convert verifiers message dicts to tau2 Message objects."""
    tau2_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            tau2_messages.append(
                Tau2UserMsg(role="user", content=msg.get("content"))
            )
        elif role == "assistant":
            tool_calls = None
            if msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    args = tc.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append(
                        Tau2ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("name", ""),
                            arguments=args,
                        )
                    )
            tau2_messages.append(
                Tau2AssistantMsg(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tool_calls or None,
                )
            )
        elif role == "tool":
            tau2_messages.append(
                Tau2ToolMsg(
                    id=msg.get("tool_call_id", ""),
                    role="tool",
                    content=msg.get("content"),
                )
            )
        # Skip system messages — tau2 evaluators don't need them
    return tau2_messages


def build_simulation_run(
    state: State,
    task_dict: dict,
) -> SimulationRun:
    """Reconstruct a tau2 SimulationRun from the verifiers rollout state."""
    all_messages: list[dict[str, Any]] = []

    # Collect all messages from the trajectory
    for step in state.get("trajectory", []):
        prompt_msgs = step.get("prompt", [])
        completion_msgs = step.get("completion", [])
        for msg in prompt_msgs:
            if isinstance(msg, dict):
                all_messages.append(msg)
            else:
                all_messages.append(msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))
        for msg in completion_msgs:
            if isinstance(msg, dict):
                all_messages.append(msg)
            else:
                all_messages.append(msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))

    # Deduplicate messages that appear in both prompt and completion of successive steps
    seen = set()
    unique_messages = []
    for msg in all_messages:
        key = (msg.get("role", ""), msg.get("content", ""), str(msg.get("tool_calls", "")))
        if key not in seen:
            seen.add(key)
            unique_messages.append(msg)

    tau2_messages = _vf_messages_to_tau2(unique_messages)

    stop_condition = state.get("stop_condition", "")
    if "max_turns" in str(stop_condition):
        termination = TerminationReason.MAX_STEPS
    elif state.get("error"):
        termination = TerminationReason.AGENT_ERROR
    else:
        termination = TerminationReason.USER_STOP

    return SimulationRun(
        id=state.get("trajectory_id", uuid.uuid4().hex),
        task_id=task_dict.get("id", "unknown"),
        start_time="",
        end_time="",
        duration=0.0,
        termination_reason=termination,
        messages=tau2_messages,
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


async def tau2_reward(state: State, info: dict, **kwargs) -> float:
    """Run tau2's full evaluation pipeline and return overall reward."""
    try:
        return await _compute_reward(state, info, return_field="reward")
    except Exception as exc:
        logger.warning("tau2_reward failed: %s", exc)
        return 0.0


async def booking_accuracy(state: State, info: dict, **kwargs) -> float:
    """Booking accuracy sub-score (tracked as metric)."""
    try:
        return await _compute_reward(state, info, return_field="booking_accuracy")
    except Exception:
        return 0.0


async def preference_satisfaction(state: State, info: dict, **kwargs) -> float:
    """Preference satisfaction sub-score (tracked as metric)."""
    try:
        return await _compute_reward(state, info, return_field="preference_satisfaction")
    except Exception:
        return 0.0


async def action_correctness(state: State, info: dict, **kwargs) -> float:
    """Action correctness sub-score (tracked as metric)."""
    try:
        return await _compute_reward(state, info, return_field="action_correctness")
    except Exception:
        return 0.0


async def _compute_reward(state: State, info: dict, return_field: str) -> float:
    """Shared implementation for reward computation.

    Caches the ``RewardInfo`` on the state to avoid recomputing for each metric.
    """
    cache_key = "__tau2_reward_info__"
    reward_info: RewardInfo | None = state.get(cache_key)

    if reward_info is None:
        task_dict = info.get("task", {})
        task = Task.model_validate(task_dict)
        simulation = build_simulation_run(state, task_dict)

        from tau2.domains.airline_a2a.environment import get_environment as get_env

        reward_info = await asyncio.to_thread(
            evaluate_simulation,
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain="airline_a2a",
            environment_constructor=get_env,
        )
        state[cache_key] = reward_info

    if return_field == "reward":
        return reward_info.reward

    breakdown = reward_info.reward_breakdown or {}
    from tau2.data_model.tasks import RewardType

    field_map = {
        "booking_accuracy": RewardType.BOOKING_ACCURACY,
        "preference_satisfaction": RewardType.PREFERENCE_SATISFACTION,
        "action_correctness": RewardType.ACTION,
    }
    reward_type = field_map.get(return_field)
    if reward_type and reward_type in breakdown:
        return breakdown[reward_type]
    return 0.0


def build_rubric() -> vf.Rubric:
    """Build the tau2 reward rubric."""
    rubric = vf.Rubric()
    rubric.add_reward_func(tau2_reward, weight=1.0)
    rubric.add_metric(booking_accuracy)
    rubric.add_metric(preference_satisfaction)
    rubric.add_metric(action_correctness)
    return rubric
