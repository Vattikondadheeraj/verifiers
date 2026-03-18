"""User agent environment: the model being trained IS the user agent.

The user agent learns to negotiate with a fixed platform (LLMAgent) to book
flights that match its preference profile.
"""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, Optional

import verifiers as vf
from verifiers.types import Messages, State

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage as Tau2AssistantMsg,
    MultiToolMessage as Tau2MultiToolMsg,
    ToolMessage as Tau2ToolMsg,
    UserMessage as Tau2UserMsg,
)
from tau2.data_model.tasks import Task
from tau2.domains.airline_a2a.data_model import FlightDB
from tau2.domains.airline_a2a.tools import AirlineTools

from .tau2_tools import (
    apply_initial_state,
    create_airline_tools,
    load_db,
)

logger = logging.getLogger(__name__)

STOP_TOKEN = "###STOP###"

USER_SYSTEM_PROMPT = """You are a personal travel assistant calling an airline's customer service to book a flight for your user.

Your goal is to get a booking that matches your user's preferences as closely as possible.

Be clear, concise, and assertive about preferences. If the agent offers something that doesn't match, negotiate for better options.

{context}"""


class Tau2UserEnv(vf.MultiTurnEnv):
    """MultiTurnEnv where the trained model plays the user agent.

    The simulated platform agent is driven by tau2's ``LLMAgent`` with a fixed LLM.
    The platform agent handles its own tool calls internally.
    """

    def __init__(
        self,
        platform_llm: str = "gpt-4.1-mini",
        platform_llm_args: Optional[dict] = None,
        db_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.platform_llm = platform_llm
        self.platform_llm_args = platform_llm_args or {}
        self.db_path = db_path
        self._base_db: Optional[FlightDB] = None

    def _get_base_db(self) -> FlightDB:
        if self._base_db is None:
            self._base_db = load_db(self.db_path)
        return self._base_db

    async def setup_state(self, state: State) -> State:
        """Initialize per-rollout state: fresh DB, tools, platform agent."""
        info = state["info"]
        if isinstance(info, str):
            info = json.loads(info)
            state["input"]["info"] = info

        task_dict = info.get("task", {})

        # Fresh DB copy per rollout
        db = deepcopy(self._get_base_db())
        tools = create_airline_tools(db)
        apply_initial_state(tools, task_dict)

        # Load policy
        from tau2.domains.airline_a2a.utils import AIRLINE_A2A_POLICY_PATH

        with open(AIRLINE_A2A_POLICY_PATH, "r") as fp:
            policy = fp.read()

        # Create platform agent (LLMAgent)
        tau2_tool_defs = list(tools.get_tools().values())
        platform_agent = LLMAgent(
            tools=tau2_tool_defs,
            domain_policy=policy,
            llm=self.platform_llm,
            llm_args=deepcopy(self.platform_llm_args),
        )

        # Get initial platform agent state
        message_history = None
        task = Task.model_validate(task_dict)
        if task.initial_state and task.initial_state.message_history:
            message_history = task.initial_state.message_history
        agent_state = platform_agent.get_init_state(message_history=message_history)

        # Store environment state
        state["__tau2_tools__"] = tools
        state["__tau2_platform_agent__"] = platform_agent
        state["__tau2_agent_state__"] = agent_state
        state["__tau2_error_count__"] = 0
        state["__tau2_max_errors__"] = 5

        # Build user context for system prompt
        user_data = info.get("user_data", {})
        context = _build_user_context(user_data)
        state["input"]["prompt"] = _prepend_system(state["prompt"], context)

        # No tool_defs for user agent (user doesn't call tools)
        state["tool_defs"] = []

        return state

    @vf.stop
    async def too_many_tool_errors(self, state: State) -> bool:
        return state.get("__tau2_error_count__", 0) >= state.get("__tau2_max_errors__", 5)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """Process the model's (user agent's) output.

        Forward the user's message to the platform agent. If the platform
        agent makes tool calls, execute them internally and keep looping
        until the platform agent produces a text response.
        """
        last_msg = _get_last_assistant(messages)
        if last_msg is None:
            return [{"role": "user", "content": "Please state your request to the airline agent."}]

        user_text = last_msg.get("content", "")

        # Check if user said STOP
        if STOP_TOKEN in user_text:
            state["final_env_response"] = [
                {"role": "user", "content": "[Conversation ended by user]"}
            ]
            return state["final_env_response"]

        platform_agent: LLMAgent = state["__tau2_platform_agent__"]
        agent_state: LLMAgentState = state["__tau2_agent_state__"]
        tools: AirlineTools = state["__tau2_tools__"]

        # Forward user message to platform agent
        user_msg = Tau2UserMsg(role="user", content=user_text)

        try:
            agent_response = await _run_platform_agent_turn(
                platform_agent=platform_agent,
                agent_state=agent_state,
                user_msg=user_msg,
                tools=tools,
                state=state,
            )
        except Exception as exc:
            logger.warning("Platform agent error: %s", exc)
            state["final_env_response"] = [
                {"role": "user", "content": f"[Platform agent error: {exc}]"}
            ]
            return state["final_env_response"]

        state["__tau2_agent_state__"] = agent_state

        # Check for platform agent stop
        if STOP_TOKEN in (agent_response or ""):
            state["final_env_response"] = [
                {"role": "user", "content": agent_response}
            ]
            return state["final_env_response"]

        # Return platform agent's text as a "user" message (from the env's perspective,
        # the platform agent's response is the environment's response to the user model)
        return [{"role": "user", "content": agent_response or ""}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_platform_agent_turn(
    platform_agent: LLMAgent,
    agent_state: LLMAgentState,
    user_msg: Tau2UserMsg,
    tools: AirlineTools,
    state: State,
) -> str:
    """Run the platform agent until it produces a text response.

    The agent may make tool calls in a loop before finally sending text.
    """
    current_msg: Any = user_msg
    max_internal_steps = 20

    for _step in range(max_internal_steps):
        agent_reply, agent_state = await asyncio.to_thread(
            platform_agent.generate_next_message, current_msg, agent_state
        )

        # If agent made tool calls, execute them and feed results back
        if agent_reply.tool_calls:
            tool_results = []
            for tc in agent_reply.tool_calls:
                try:
                    result = tools.use_tool(tc.name, **tc.arguments)
                    if not isinstance(result, str):
                        result = json.dumps(result, default=str)
                    tool_results.append(
                        Tau2ToolMsg(
                            id=tc.id,
                            role="tool",
                            content=result,
                        )
                    )
                except Exception as exc:
                    state["__tau2_error_count__"] = state.get("__tau2_error_count__", 0) + 1
                    tool_results.append(
                        Tau2ToolMsg(
                            id=tc.id,
                            role="tool",
                            content=f"Error: {exc}",
                            error=True,
                        )
                    )

            if len(tool_results) == 1:
                current_msg = tool_results[0]
            else:
                current_msg = Tau2MultiToolMsg(role="tool", tool_messages=tool_results)
            continue

        # Agent produced text
        return agent_reply.content or ""

    return "[Platform agent exceeded maximum internal steps]"


def _get_last_assistant(messages: Messages) -> Optional[dict[str, Any]]:
    """Find the last assistant message in the conversation."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg
        if hasattr(msg, "role") and msg.role == "assistant":
            return msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
    return None


def _build_user_context(user_data: Optional[dict]) -> str:
    """Build context string from user data for the system prompt."""
    if not user_data:
        return ""

    parts = []
    emails = user_data.get("emails", [])
    if emails:
        email_text = "\n\n".join(
            f"Subject: {e.get('subject', '')}\n{e.get('body', '')}" for e in emails[:5]
        )
        parts.append(f"<email_history>\n{email_text}\n</email_history>")

    profile = user_data.get("preference_profile")
    if profile:
        parts.append(f"<preference_profile>\n{json.dumps(profile, indent=2)}\n</preference_profile>")

    reason = user_data.get("reason_for_call")
    if reason:
        parts.append(f"<reason_for_call>\n{reason}\n</reason_for_call>")

    return "\n\n".join(parts)


def _prepend_system(prompt: Messages, context: str) -> Messages:
    """Prepend system message with user context to the prompt."""
    system_msg = {
        "role": "system",
        "content": USER_SYSTEM_PROMPT.format(context=context),
    }
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        return [system_msg] + list(prompt[1:])
    return [system_msg] + list(prompt)
