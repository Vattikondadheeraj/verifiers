"""Tests for sliding window context management and full-history reward evaluation."""

import pytest
from verifiers.types import AssistantMessage

from tau2_bench.tau2_platform_env import _msgs_to_dicts, _sliding_window
from tau2_bench.tau2_rewards import build_simulation_run


# ---------------------------------------------------------------------------
# _msgs_to_dicts
# ---------------------------------------------------------------------------


def test_msgs_to_dicts_plain_dicts():
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    result = _msgs_to_dicts(msgs)
    assert result == msgs


def test_msgs_to_dicts_typed_objects():
    msg = AssistantMessage(role="assistant", content="hi", tool_calls=None)
    result = _msgs_to_dicts([msg])
    assert isinstance(result[0], dict)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "hi"


def test_msgs_to_dicts_mixed():
    msgs = [
        {"role": "user", "content": "hello"},
        AssistantMessage(role="assistant", content="hi", tool_calls=None),
    ]
    result = _msgs_to_dicts(msgs)
    assert all(isinstance(m, dict) for m in result)
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# _sliding_window
# ---------------------------------------------------------------------------


def _make_messages(n_middle: int, content_chars: int = 10) -> list[dict]:
    """Build a message list: system + n_middle user/assistant pairs + final user msg."""
    msgs = [{"role": "system", "content": "S" * content_chars}]
    for i in range(n_middle):
        msgs.append({"role": "user", "content": f"U{i}" + "x" * content_chars})
        msgs.append({"role": "assistant", "content": f"A{i}" + "x" * content_chars})
    msgs.append({"role": "user", "content": "final user message"})
    return msgs


def test_sliding_window_short_conversation_unchanged():
    msgs = _make_messages(3, content_chars=10)
    result = _sliding_window(msgs, seq_len=25000)
    assert result == msgs


def test_sliding_window_always_keeps_system_and_last():
    # Build a very long conversation that will definitely exceed the budget
    msgs = _make_messages(n_middle=200, content_chars=200)
    result = _sliding_window(msgs, seq_len=2000)
    assert result[0]["role"] == "system"
    assert result[-1] == msgs[-1]


def test_sliding_window_drops_oldest_first():
    msgs = _make_messages(n_middle=200, content_chars=200)
    result = _sliding_window(msgs, seq_len=2000)
    # The dropped messages should be the oldest ones (indices 1, 2, ...)
    # Result should be a contiguous tail of the original middle messages
    if len(result) < len(msgs):
        # The kept middle messages should match the END of the original middle
        kept_middle = result[1:-1]
        original_middle = msgs[1:-1]
        tail = original_middle[len(original_middle) - len(kept_middle):]
        assert kept_middle == tail


def test_sliding_window_only_2_messages_unchanged():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    result = _sliding_window(msgs, seq_len=100)
    assert result == msgs


def _make_messages_with_tool_calls(n_rounds: int, content_chars: int = 200) -> list[dict]:
    """Build: system + n_rounds of (user, assistant+tool_calls, tool, assistant_text) + final user."""
    msgs = [{"role": "system", "content": "S" * content_chars}]
    for i in range(n_rounds):
        msgs.append({"role": "user", "content": f"U{i}" + "x" * content_chars})
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"tc{i}", "name": "search_flights", "arguments": "{}"}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"result{i}" + "x" * content_chars})
        msgs.append({"role": "assistant", "content": f"A{i}" + "x" * content_chars, "tool_calls": None})
    msgs.append({"role": "user", "content": "final user message"})
    return msgs


def test_sliding_window_drops_tool_call_and_result_atomically():
    """When dropping due to budget, assistant+tool_calls and its tool results drop together."""
    msgs = _make_messages_with_tool_calls(n_rounds=50, content_chars=200)
    result = _sliding_window(msgs, seq_len=5000)

    # No orphaned tool result: every tool message must have a preceding assistant with tool_calls
    for i, msg in enumerate(result):
        if msg.get("role") == "tool":
            # Previous message must be either another tool msg or an assistant with tool_calls
            prev = result[i - 1]
            assert prev.get("role") in ("tool", "assistant") and (
                prev.get("tool_calls") or prev.get("role") == "tool"
            ), f"Orphaned tool message at index {i}: prev role={prev.get('role')}"

    # No dangling assistant+tool_calls without a following tool result
    for i, msg in enumerate(result[:-1]):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            nxt = result[i + 1]
            assert nxt.get("role") == "tool", (
                f"assistant+tool_calls at index {i} not followed by tool result, got role={nxt.get('role')}"
            )


def test_sliding_window_result_fits_budget():
    msgs = _make_messages(n_middle=100, content_chars=500)
    seq_len = 5000
    result = _sliding_window(msgs, seq_len=seq_len)
    budget = seq_len - 8192
    if budget > 0:
        # Token estimate of result should be <= budget
        total_chars = sum(
            len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
            for m in result
        )
        token_estimate = total_chars // 4
        assert token_estimate <= budget or len(result) == 2  # at minimum system + last


# ---------------------------------------------------------------------------
# __full_messages__ in state / build_simulation_run
# ---------------------------------------------------------------------------


def _make_state_with_full_messages(messages: list[dict]) -> dict:
    return {
        "__full_messages__": messages,
        "trajectory": [],
        "trajectory_id": "test-id",
    }


def _make_state_with_platform_msgs(platform_msgs: list) -> dict:
    return {
        "__platform_msgs__": platform_msgs,
        "trajectory": [],
        "trajectory_id": "test-id",
    }


def test_build_simulation_run_uses_full_messages_for_platform():
    """When __full_messages__ is set, build_simulation_run uses it (not trajectory)."""
    from tau2.data_model.message import AssistantMessage as Tau2Asst, UserMessage as Tau2User

    full_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I want to book a flight."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "search_direct_flights", "arguments": '{"origin": "JFK"}'}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"flights": []}'},
        {"role": "assistant", "content": "No flights found."},
        {"role": "user", "content": "OK thanks."},
    ]

    state = _make_state_with_full_messages(full_msgs)
    task_dict = {"id": "t1", "evaluation_criteria": None, "user_scenario": {}, "initial_state": None}
    sim = build_simulation_run(state, task_dict)

    # Should have user + assistant + tool + assistant + user messages (system skipped)
    roles = [m.role for m in sim.messages]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles
    # System messages are skipped by _vf_messages_to_tau2
    assert "system" not in roles


def test_build_simulation_run_strips_trailing_tool_call():
    """Trailing assistant tool-call with no result is stripped."""
    full_msgs = [
        {"role": "user", "content": "Book me a flight."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "book_reservation", "arguments": "{}"}],
        },
        # No tool result — rollout ended mid-call
    ]

    state = _make_state_with_full_messages(full_msgs)
    task_dict = {"id": "t1", "evaluation_criteria": None, "user_scenario": {}, "initial_state": None}
    sim = build_simulation_run(state, task_dict)

    # Trailing tool call should be stripped
    assert all(
        not (m.role == "assistant" and m.tool_calls)
        for m in sim.messages
    )


def test_build_simulation_run_prefers_platform_msgs_for_user_training():
    """When __platform_msgs__ is present, it takes priority over __full_messages__."""
    from tau2.data_model.message import UserMessage as Tau2User

    platform_msgs = [Tau2User(role="user", content="platform user msg")]
    state = {
        "__platform_msgs__": platform_msgs,
        "__full_messages__": [{"role": "user", "content": "trained model msg"}],
        "trajectory": [],
        "trajectory_id": "test-id",
    }
    task_dict = {"id": "t1", "evaluation_criteria": None, "user_scenario": {}, "initial_state": None}
    sim = build_simulation_run(state, task_dict)

    # Should use platform_msgs, not __full_messages__
    assert sim.messages[0].content == "platform user msg"


def test_build_simulation_run_falls_back_to_trajectory():
    """When neither __platform_msgs__ nor __full_messages__ is set, use trajectory."""
    state = {
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "completion": [{"role": "assistant", "content": "hi"}],
            }
        ],
        "trajectory_id": "test-id",
    }
    task_dict = {"id": "t1", "evaluation_criteria": None, "user_scenario": {}, "initial_state": None}
    sim = build_simulation_run(state, task_dict)
    roles = [m.role for m in sim.messages]
    assert "user" in roles
    assert "assistant" in roles
