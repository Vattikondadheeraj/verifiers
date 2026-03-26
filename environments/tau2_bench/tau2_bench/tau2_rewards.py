"""Reward system wrapping tau2's evaluation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from copy import deepcopy
from pathlib import Path
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
    """Reconstruct a tau2 SimulationRun from the verifiers rollout state.

    For user training, the platform agent's messages (including book_reservation
    tool calls) are stored in ``state["__platform_msgs__"]`` and used directly.
    For platform training, the verifiers trajectory is used.
    """
    explicit = state.get("__tau2_termination__")
    stop_condition = state.get("stop_condition", "")
    if explicit == "user_stop":
        termination = TerminationReason.USER_STOP
    elif explicit == "agent_stop":
        termination = TerminationReason.AGENT_STOP
    elif explicit == "error" or state.get("error"):
        termination = TerminationReason.AGENT_ERROR
    elif "max_turns" in str(stop_condition):
        termination = TerminationReason.MAX_STEPS
    elif state.get("__tau2_booking_confirmed__"):
        # Booking was confirmed but no explicit stop signal — treat as USER_STOP
        # so the tau2 evaluator scores the interaction (it requires USER_STOP or AGENT_STOP).
        termination = TerminationReason.USER_STOP
    else:
        # Fallback: unknown stop condition → treat as agent error to avoid
        # incorrect reward for truncated/failed rollouts
        termination = TerminationReason.AGENT_ERROR

    platform_msgs = state.get("__platform_msgs__")
    if platform_msgs:
        # User training: use the platform agent's own message history directly.
        # These are tau2 Message objects containing the real tool calls (including
        # book_reservation) that the evaluators need to score the interaction.
        tau2_messages = list(platform_msgs)
    else:
        # Platform training: reconstruct from the verifiers trajectory.
        all_messages: list[dict[str, Any]] = []
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

        # Deduplicate messages that appear in both prompt and completion of successive steps.
        # tool_call_id must be included in the key so that two different tool results with
        # identical content are not incorrectly collapsed into one.
        seen = set()
        unique_messages = []
        for msg in all_messages:
            key = (
                msg.get("role", ""),
                msg.get("content", ""),
                str(msg.get("tool_calls", "")),
                msg.get("tool_call_id", ""),
            )
            if key not in seen:
                seen.add(key)
                unique_messages.append(msg)

        # Strip trailing assistant tool-call messages that have no following tool result.
        # This happens when a rollout ends (max_turns) mid-tool-call. The tau2 evaluator
        # crashes if it sees a tool_call with no subsequent ToolMessage.
        while unique_messages:
            last = unique_messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                unique_messages.pop()
            else:
                break

        tau2_messages = _vf_messages_to_tau2(unique_messages)

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


def _serialize_tau2_messages(messages: list) -> list[dict]:
    """Convert tau2 Message objects to plain dicts."""
    result = []
    for m in messages:
        if hasattr(m, "model_dump"):
            d = m.model_dump()
        elif isinstance(m, dict):
            d = m
        else:
            d = dict(m)
        # Keep only the fields useful for visualization
        entry = {"role": d.get("role", ""), "content": d.get("content", "")}
        if d.get("tool_calls"):
            entry["tool_calls"] = d["tool_calls"]
        result.append(entry)
    return result


def _save_sidecar(state: State, reward: float, sidecar_dir: Path):
    """Write a sidecar JSON capturing the exact LLM conversations for both agents."""
    trajectory_id = state.get("trajectory_id", uuid.uuid4().hex)
    sidecar_path = sidecar_dir / f"{trajectory_id}.json"
    if sidecar_path.exists():
        return  # already written by a prior reward function call for this rollout

    # Trained model's conversation (from verifiers trajectory)
    trained_msgs: list[dict] = []
    seen = set()
    for step in state.get("trajectory", []):
        for msg in list(step.get("prompt", [])) + list(step.get("completion", [])):
            d = msg if isinstance(msg, dict) else (msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))
            role = d.get("role", "")
            # Normalize verifiers TextMessage (role="text") → role="user".
            # TextMessage carries the JSON-encoded rollout input; unwrap its content.
            if role == "text":
                content = d.get("content", "")
                if content.startswith("[{"):
                    try:
                        import json as _json
                        inner = _json.loads(content)
                        if isinstance(inner, list) and inner:
                            content = inner[-1].get("content", content)
                    except Exception:
                        pass
                d = {"role": "user", "content": content}
            key = (d.get("role", ""), d.get("content", ""), str(d.get("tool_calls", "")), d.get("tool_call_id", ""))
            if key not in seen:
                seen.add(key)
                trained_msgs.append(d)

    # Counterpart's exact conversation
    counterpart_system_prompt = ""
    counterpart_msgs: list[dict] = []

    user_state = state.get("__tau2_user_state__")
    user_sim = state.get("__tau2_user_sim__")
    if user_state is not None and user_sim is not None:
        # Platform training: counterpart is user simulator
        # user_state.flip_roles() gives the exact messages sent to the user sim's LLM
        counterpart_system_prompt = user_sim.system_prompt
        counterpart_msgs = _serialize_tau2_messages(user_state.flip_roles())

    platform_msgs = state.get("__platform_msgs__")
    if platform_msgs:
        # User training: counterpart is platform simulator
        agent = state.get("__tau2_platform_agent__")
        if agent is not None:
            counterpart_system_prompt = agent.system_prompt
        counterpart_msgs = _serialize_tau2_messages(platform_msgs)

    # Extract reward breakdown from cached reward_info if available
    reward_breakdown = {}
    for key, val in state.items():
        if key.startswith("__tau2_reward_info__") and val is not None:
            rb = getattr(val, "reward_breakdown", None) or {}
            for rt, score in rb.items():
                reward_breakdown[rt.value] = round(float(score), 4)
            break

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "trajectory_id": trajectory_id,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "termination": state.get("__tau2_termination__", "unknown"),
        "num_turns": state.get("__tau2_turn_count__", None),
        "trained_model_messages": trained_msgs,
        "counterpart_system_prompt": counterpart_system_prompt,
        "counterpart_messages": counterpart_msgs,
    }
    sidecar_path.write_text(json.dumps(record))


def _make_reward_fn(return_field: str, reward_basis: list[str] | None, sidecar_dir: Path | None = None):
    """Create a reward function closed over return_field, reward_basis, and sidecar_dir."""
    async def _fn(state: State, info: dict, **kwargs) -> float:
        try:
            result = await _compute_reward(state, info, return_field=return_field, reward_basis=reward_basis)
            if return_field == "reward" and sidecar_dir is not None:
                await asyncio.to_thread(_save_sidecar, state, result, sidecar_dir)
            return result
        except Exception as exc:
            logger.warning("tau2_%s failed: %s", return_field, exc)
            return 0.0
    _fn.__name__ = return_field
    return _fn


async def _compute_reward(state: State, info: dict, return_field: str, reward_basis: list[str] | None = None) -> float:
    """Shared implementation for reward computation.

    Caches the ``RewardInfo`` on the state to avoid recomputing for each metric.
    """
    cache_key = f"__tau2_reward_info__{','.join(sorted(reward_basis)) if reward_basis else 'default'}__"
    reward_info: RewardInfo | None = state.get(cache_key)

    if reward_info is None:
        task_dict = info.get("task", {})
        task = Task.model_validate(task_dict)
        if reward_basis is not None and task.evaluation_criteria is not None:
            from tau2.data_model.tasks import RewardType
            task.evaluation_criteria.reward_basis = [RewardType(r) for r in reward_basis]
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

    if return_field == "success_rate":
        if RewardType.DB in breakdown:
            return 1.0 if breakdown[RewardType.DB] >= 1.0 else 0.0
        # User training: DB not evaluated — use booking_accuracy as proxy
        ba_score = breakdown.get(RewardType.BOOKING_ACCURACY, 0.0)
        return 1.0 if ba_score >= 1.0 else 0.0

    field_map = {
        "booking_accuracy": RewardType.BOOKING_ACCURACY,
        "preference_satisfaction": RewardType.PREFERENCE_SATISFACTION,
        "action_correctness": RewardType.ACTION,
    }
    reward_type = field_map.get(return_field)
    if reward_type and reward_type in breakdown:
        return breakdown[reward_type]
    return 0.0


def build_rubric(reward_basis: list[str] | None = None, sidecar_dir: str | None = None) -> vf.Rubric:
    """Build the tau2 reward rubric.

    Args:
        reward_basis: List of reward types to optimize, e.g.
            ``["DB", "ACTION", "BOOKING_ACCURACY", "PREFERENCE_SATISFACTION"]``.
            If None, uses whatever is stored in each task's evaluation_criteria.
        sidecar_dir: If set, write a JSON sidecar per rollout with the exact LLM
            conversations for both the trained model and its counterpart agent.
    """
    sd = Path(sidecar_dir) if sidecar_dir else None
    rubric = vf.Rubric()
    rubric.add_reward_func(_make_reward_fn("reward", reward_basis, sidecar_dir=sd), weight=1.0)
    rubric.add_metric(_make_reward_fn("booking_accuracy", reward_basis))
    rubric.add_metric(_make_reward_fn("preference_satisfaction", reward_basis))
    rubric.add_metric(_make_reward_fn("action_correctness", reward_basis))
    rubric.add_metric(_make_reward_fn("success_rate", reward_basis))
    return rubric
