"""Reward system wrapping tau2's evaluation pipeline.

Supports all reward types:
  - DB, ACTION, COMMUNICATE, NL_ASSERTION (upstream)
  - BOOKING_ACCURACY, PREFERENCE_SATISFACTION (airline_a2a)
  - PLATFORM_REVENUE, POLICY_COMPLIANCE, DATA_LEAKAGE (marketplace)

Per-agent rewards:
  - user_reward  = PREFERENCE_SATISFACTION score
  - platform_reward = POLICY_COMPLIANCE score
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
from tau2.data_model.tasks import RewardType, Task
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
                    # Handle both flat (name/arguments) and nested (function.name/function.arguments) formats
                    fn = tc.get("function")
                    if fn:
                        name = fn.get("name", "")
                        args = fn.get("arguments", "{}")
                    else:
                        name = tc.get("name", "")
                        args = tc.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append(
                        Tau2ToolCall(
                            id=tc.get("id", ""),
                            name=name,
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
    # Always evaluate every trace regardless of termination reason.
    # tau2's evaluator hard-gates on USER_STOP/AGENT_STOP, so we map everything
    # to USER_STOP to get partial rewards even on failed traces.
    termination = TerminationReason.USER_STOP

    platform_msgs = state.get("__platform_msgs__")
    if platform_msgs:
        # User training: use the platform agent's own message history directly.
        tau2_messages = list(platform_msgs)
    elif state.get("__full_messages__"):
        # Platform training: use the directly tracked full message history.
        all_messages: list[dict[str, Any]] = list(state["__full_messages__"])

        # Strip trailing assistant tool-call messages that have no following tool result.
        while all_messages:
            last = all_messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                all_messages.pop()
            else:
                break

        tau2_messages = _vf_messages_to_tau2(all_messages)
    else:
        # Fallback: reconstruct from the verifiers trajectory.
        all_messages = []
        for step in state.get("trajectory", []):
            prompt_msgs = step.get("prompt", [])
            completion_msgs = step.get("completion", [])
            for msg in prompt_msgs + completion_msgs:
                if isinstance(msg, dict):
                    all_messages.append(msg)
                else:
                    all_messages.append(msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))

        def _norm_content(c):
            s = (c or "").strip()
            s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL).strip()
            return s

        seen = set()
        unique_messages = []
        for msg in all_messages:
            key = (
                msg.get("role", ""),
                _norm_content(msg.get("content")),
                str(msg.get("tool_calls") or ""),
                msg.get("tool_call_id", ""),
            )
            if key not in seen:
                seen.add(key)
                unique_messages.append(msg)

        while unique_messages:
            last = unique_messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                unique_messages.pop()
            else:
                break

        tau2_messages = _vf_messages_to_tau2(unique_messages)

    # Include marketplace info if available (booked_airline tracking)
    marketplace_info = state.get("__marketplace_info__")

    return SimulationRun(
        id=state.get("trajectory_id", uuid.uuid4().hex),
        task_id=task_dict.get("id", "unknown"),
        start_time="",
        end_time="",
        duration=0.0,
        termination_reason=termination,
        messages=tau2_messages,
        marketplace_info=marketplace_info,
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
        return

    trained_msgs: list[dict] = []
    seen = set()
    for step in state.get("trajectory", []):
        for msg in list(step.get("prompt", [])) + list(step.get("completion", [])):
            d = msg if isinstance(msg, dict) else (msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))
            role = d.get("role", "")
            if role == "text":
                content = d.get("content", "")
                if content.startswith("[{"):
                    try:
                        inner = json.loads(content)
                        if isinstance(inner, list) and inner:
                            content = inner[-1].get("content", content)
                    except Exception:
                        pass
                d = {"role": "user", "content": content}
            key = (d.get("role", ""), d.get("content", ""), str(d.get("tool_calls", "")), d.get("tool_call_id", ""))
            if key not in seen:
                seen.add(key)
                trained_msgs.append(d)

    counterpart_system_prompt = ""
    counterpart_msgs: list[dict] = []

    user_state = state.get("__tau2_user_state__")
    user_sim = state.get("__tau2_user_sim__")
    if user_state is not None and user_sim is not None:
        counterpart_system_prompt = user_sim.system_prompt
        counterpart_msgs = _serialize_tau2_messages(user_state.flip_roles())

    platform_msgs = state.get("__platform_msgs__")
    if platform_msgs:
        agent = state.get("__tau2_platform_agent__")
        if agent is not None:
            counterpart_system_prompt = agent.system_prompt
        counterpart_msgs = _serialize_tau2_messages(platform_msgs)

    reward_breakdown = {}
    for key, val in state.items():
        if key.startswith("__tau2_reward_info__") and val is not None:
            rb = getattr(val, "reward_breakdown", None) or {}
            for rt, score in rb.items():
                reward_breakdown[rt.value if hasattr(rt, "value") else str(rt)] = round(float(score), 4)
            break

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "trajectory_id": trajectory_id,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "termination": state.get("__tau2_termination__", "unknown"),
        "num_turns": state.get("__tau2_turn_count__", None),
        "booked_airline": state.get("__booked_airline__"),
        "trained_model_messages": trained_msgs,
        "counterpart_system_prompt": counterpart_system_prompt,
        "counterpart_messages": counterpart_msgs,
    }
    sidecar_path.write_text(json.dumps(record))


def _make_reward_fn(
    return_field: str,
    reward_basis: list[str] | None,
    sidecar_dir: Path | None = None,
    enable_leakage: bool = False,
):
    """Create a reward function closed over return_field, reward_basis, and sidecar_dir."""
    async def _fn(state: State, info: dict, **kwargs) -> float:
        try:
            result = await _compute_reward(
                state, info,
                return_field=return_field,
                reward_basis=reward_basis,
                enable_leakage=enable_leakage,
            )
            if return_field == "reward" and sidecar_dir is not None:
                await asyncio.to_thread(_save_sidecar, state, result, sidecar_dir)
            return result
        except Exception as exc:
            import traceback as _tb
            logger.error(
                "tau2_%s failed for trajectory %s: %s\n%s",
                return_field,
                state.get("trajectory_id", "unknown"),
                exc,
                _tb.format_exc(),
            )
            state["__reward_error__"] = f"{type(exc).__name__}: {exc}"
            return 0.0
    _fn.__name__ = return_field
    return _fn


async def _reward_error_metric(state: State, info: dict, **kwargs) -> float:
    """1.0 if reward computation failed for this rollout, 0.0 otherwise."""
    return 1.0 if state.get("__reward_error__") else 0.0


async def _compute_reward(
    state: State,
    info: dict,
    return_field: str,
    reward_basis: list[str] | None = None,
    enable_leakage: bool = False,
) -> float:
    """Shared implementation for reward computation.

    Caches the ``RewardInfo`` on the state to avoid recomputing for each metric.
    """
    cache_key = f"__tau2_reward_info__{','.join(sorted(reward_basis)) if reward_basis else 'default'}__"
    reward_info: RewardInfo | None = state.get(cache_key)

    if reward_info is None:
        task_dict = info.get("task", {})
        task = Task.model_validate(task_dict)
        if reward_basis is not None and task.evaluation_criteria is not None:
            task.evaluation_criteria.reward_basis = [RewardType(r) for r in reward_basis]
        simulation = build_simulation_run(state, task_dict)

        from tau2.domains.airline_a2a.environment import get_environment as get_env

        # Choose evaluation type based on what's needed
        if enable_leakage and RewardType.DATA_LEAKAGE in set(task.evaluation_criteria.reward_basis or []):
            eval_type = EvaluationType.ALL_WITH_LEAKAGE
        else:
            eval_type = EvaluationType.ALL

        # Pass booked_airline and flight_db for REVENUE and POLICY_COMPLIANCE evaluators
        eval_kwargs: dict[str, Any] = {}
        booked_airline = state.get("__booked_airline__")
        if booked_airline:
            eval_kwargs["booked_airline"] = booked_airline
        flight_db = state.get("__flight_db__")
        if flight_db:
            eval_kwargs["flight_db"] = flight_db

        reward_info = await asyncio.to_thread(
            evaluate_simulation,
            simulation=simulation,
            task=task,
            evaluation_type=eval_type,
            solo_mode=False,
            domain="airline_a2a",
            environment_constructor=get_env,
            **eval_kwargs,
        )
        state[cache_key] = reward_info

    if return_field == "reward":
        return reward_info.reward

    # Per-agent rewards
    if return_field == "user_reward":
        return reward_info.user_reward if reward_info.user_reward is not None else 0.0
    if return_field == "platform_reward":
        return reward_info.platform_reward if reward_info.platform_reward is not None else 0.0

    breakdown = reward_info.reward_breakdown or {}

    if return_field == "success_rate":
        if RewardType.DB in breakdown:
            return 1.0 if breakdown[RewardType.DB] >= 1.0 else 0.0
        ba_score = breakdown.get(RewardType.BOOKING_ACCURACY, 0.0)
        return 1.0 if ba_score >= 1.0 else 0.0

    field_map = {
        "booking_accuracy": RewardType.BOOKING_ACCURACY,
        "preference_satisfaction": RewardType.PREFERENCE_SATISFACTION,
        "action_correctness": RewardType.ACTION,
        "platform_revenue": RewardType.PLATFORM_REVENUE,
        "policy_compliance": RewardType.POLICY_COMPLIANCE,
        "data_leakage": RewardType.DATA_LEAKAGE,
        "db": RewardType.DB,
        "communicate": RewardType.COMMUNICATE,
    }
    reward_type = field_map.get(return_field)
    if reward_type and reward_type in breakdown:
        return breakdown[reward_type]
    return 0.0


def build_rubric(
    reward_basis: list[str] | None = None,
    sidecar_dir: str | None = None,
    enable_leakage: bool = False,
) -> vf.Rubric:
    """Build the tau2 reward rubric.

    Args:
        reward_basis: List of reward types to optimize, e.g.
            ``["BOOKING_ACCURACY", "PREFERENCE_SATISFACTION"]`` for single-airline,
            ``["PREFERENCE_SATISFACTION", "POLICY_COMPLIANCE"]`` for marketplace.
            If None, uses whatever is stored in each task's evaluation_criteria.
        sidecar_dir: If set, write a JSON sidecar per rollout.
        enable_leakage: If True, run the DATA_LEAKAGE evaluator (requires LLM judge,
            adds latency). Only enable for eval, not training.
    """
    sd = Path(sidecar_dir) if sidecar_dir else None

    rubric = vf.Rubric()

    # Main reward (weighted average of active evaluators per reward_basis)
    rubric.add_reward_func(
        _make_reward_fn("reward", reward_basis, sidecar_dir=sd, enable_leakage=enable_leakage),
        weight=1.0,
    )

    # Per-agent rewards (for co-training: each agent optimizes its own reward)
    rubric.add_metric(_make_reward_fn("user_reward", reward_basis, enable_leakage=enable_leakage))
    rubric.add_metric(_make_reward_fn("platform_reward", reward_basis, enable_leakage=enable_leakage))

    # Individual evaluator scores as metrics (logged to wandb)
    rubric.add_metric(_make_reward_fn("booking_accuracy", reward_basis))
    rubric.add_metric(_make_reward_fn("preference_satisfaction", reward_basis))
    rubric.add_metric(_make_reward_fn("platform_revenue", reward_basis))
    rubric.add_metric(_make_reward_fn("policy_compliance", reward_basis))
    rubric.add_metric(_make_reward_fn("action_correctness", reward_basis))
    rubric.add_metric(_make_reward_fn("success_rate", reward_basis))
    if enable_leakage:
        rubric.add_metric(_make_reward_fn("data_leakage", reward_basis, enable_leakage=True))

    # Error tracking
    rubric.add_metric(_reward_error_metric)

    return rubric
