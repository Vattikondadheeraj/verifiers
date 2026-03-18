"""Platform agent environment: the model being trained IS the platform (customer service) agent."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, Optional

import verifiers as vf
from verifiers.types import Messages, State

from tau2.data_model.message import (
    AssistantMessage as Tau2AssistantMsg,
    UserMessage as Tau2UserMsg,
)
from tau2.data_model.tasks import Task
from tau2.domains.airline_a2a.data_model import FlightDB
from tau2.domains.airline_a2a.tools import AirlineTools
from tau2.user.user_simulator import UserSimulator

from .tau2_tools import (
    apply_initial_state,
    create_airline_tools,
    execute_tool_call,
    get_vf_tool_defs,
    load_db,
)

logger = logging.getLogger(__name__)

STOP_TOKEN = "###STOP###"
TRANSFER_TOKEN = "###TRANSFER###"

PLATFORM_SYSTEM_PROMPT = """You are a customer service agent for an airline. Help the user with their request according to the policy below.

In each turn you can either:
- Send a text message to the user.
- Make one or more tool calls.
You cannot do both at the same time.

Always follow the policy. Always generate valid JSON for tool calls.

<policy>
{policy}
</policy>"""


class Tau2PlatformEnv(vf.MultiTurnEnv):
    """MultiTurnEnv where the trained model plays the platform (customer service) agent.

    The simulated user is driven by tau2's ``UserSimulator`` with a fixed LLM.
    Tool calls from the trained model are executed against ``AirlineTools``.
    """

    def __init__(
        self,
        user_llm: str = "gpt-4.1-mini",
        user_llm_args: Optional[dict] = None,
        db_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {}
        self.db_path = db_path
        self._base_db: Optional[FlightDB] = None

    def _get_base_db(self) -> FlightDB:
        if self._base_db is None:
            self._base_db = load_db(self.db_path)
        return self._base_db

    async def setup_state(self, state: State) -> State:
        """Initialize per-rollout state: fresh DB, tools, user simulator."""
        info = state["info"]
        if isinstance(info, str):
            info = json.loads(info)
            state["input"]["info"] = info

        task_dict = info.get("task", {})

        # Fresh DB copy per rollout
        db = deepcopy(self._get_base_db())
        tools = create_airline_tools(db)
        apply_initial_state(tools, task_dict)

        # Build verifiers tool defs for the model
        vf_tool_defs = get_vf_tool_defs(tools)
        state["tool_defs"] = vf_tool_defs

        # Policy for system prompt
        from tau2.domains.airline_a2a.utils import AIRLINE_A2A_POLICY_PATH

        with open(AIRLINE_A2A_POLICY_PATH, "r") as fp:
            policy = fp.read()

        # Store environment state
        state["__tau2_tools__"] = tools
        state["__tau2_policy__"] = policy
        state["__tau2_error_count__"] = 0
        state["__tau2_max_errors__"] = 5
        state["__tau2_turn_count__"] = 0
        state["__tau2_min_turns__"] = 3

        # Create simulated user
        task = Task.model_validate(task_dict)
        user_instructions = task.user_scenario.instructions
        user_sim = UserSimulator(
            instructions=user_instructions,
            llm=self.user_llm,
            llm_args=deepcopy(self.user_llm_args),
        )

        # Get initial user state from any message history
        message_history = None
        if task.initial_state and task.initial_state.message_history:
            message_history = task.initial_state.message_history

        user_state = user_sim.get_init_state(message_history=message_history)
        state["__tau2_user_sim__"] = user_sim
        state["__tau2_user_state__"] = user_state

        # Save both system prompts for visualization
        state["platform_system_prompt"] = PLATFORM_SYSTEM_PROMPT.format(policy=policy)
        state["user_system_prompt"] = user_sim.system_prompt

        # Override system prompt
        state["input"]["prompt"] = _prepend_system(
            state["prompt"], policy
        )

        return state

    @vf.stop
    async def too_many_tool_errors(self, state: State) -> bool:
        return state.get("__tau2_error_count__", 0) >= state.get("__tau2_max_errors__", 5)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """Process the model's (platform agent's) output.

        Two cases:
        1. Model made tool calls -> execute them, return tool results.
        2. Model sent text -> forward to simulated user, return user reply.
        """
        last_msg = _get_last_assistant(messages)
        if last_msg is None:
            return [{"role": "user", "content": "Hello, I need help with a booking."}]

        tools: AirlineTools = state["__tau2_tools__"]
        tool_calls = last_msg.get("tool_calls")

        # Case 1: Tool calls
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("arguments", "{}")
                result = await execute_tool_call(tools, tc_name, tc_args)
                if result.startswith("Error:"):
                    state["__tau2_error_count__"] = state.get("__tau2_error_count__", 0) + 1
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    }
                )
            return tool_results

        # Case 2: Text response -> forward to simulated user
        agent_text = last_msg.get("content", "")
        user_sim: UserSimulator = state["__tau2_user_sim__"]
        user_state = state["__tau2_user_state__"]

        # Increment turn count (each text exchange = 1 turn)
        state["__tau2_turn_count__"] = state.get("__tau2_turn_count__", 0) + 1

        agent_msg = Tau2AssistantMsg(role="assistant", content=agent_text)
        try:
            user_msg, user_state = await asyncio.to_thread(
                user_sim.generate_next_message, agent_msg, user_state
            )
        except Exception as exc:
            logger.warning("User simulator error: %s", exc)
            state["final_env_response"] = [
                {"role": "user", "content": f"[User simulator error: {exc}] ###STOP###"}
            ]
            return state["final_env_response"]

        state["__tau2_user_state__"] = user_state

        user_content = user_msg.content or ""

        # Check for stop signals, but enforce minimum turns
        min_turns = state.get("__tau2_min_turns__", 3)
        current_turn = state.get("__tau2_turn_count__", 0)
        if UserSimulator.is_stop(user_msg):
            if current_turn >= min_turns:
                state["final_env_response"] = [
                    {"role": "user", "content": user_content}
                ]
                return state["final_env_response"]
            else:
                # Strip stop tokens and continue the conversation
                stripped = user_content
                for token in ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]:
                    stripped = stripped.replace(token, "").strip()
                if not stripped:
                    stripped = "Please continue — I need more help with this."
                return [{"role": "user", "content": stripped}]

        return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_last_assistant(messages: Messages) -> Optional[dict[str, Any]]:
    """Find the last assistant message in the conversation."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg
        if hasattr(msg, "role") and msg.role == "assistant":
            return msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
    return None


def _prepend_system(prompt: Messages, policy: str) -> Messages:
    """Prepend system message with policy to the prompt."""
    system_msg = {
        "role": "system",
        "content": PLATFORM_SYSTEM_PROMPT.format(policy=policy),
    }
    # If prompt already has a system message, replace it
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        return [system_msg] + list(prompt[1:])
    return [system_msg] + list(prompt)
