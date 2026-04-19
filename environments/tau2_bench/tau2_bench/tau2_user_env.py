"""User agent environment: the model being trained IS the user agent.

Two modes:
  - Single-airline: user sends text to a fixed platform LLMAgent
  - Marketplace: user makes tool calls (list_airlines, query_airline, get_user_details)
    to shop across multiple airline agents
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
from tau2.environment.environment import Environment as Tau2Environment

from .tau2_tools import (
    apply_initial_state,
    create_airline_tools,
    execute_tool_call,
    get_marketplace_tool_defs,
    load_db,
)

logger = logging.getLogger(__name__)

STOP_TOKEN = "###STOP###"

def _load_user_guidelines() -> str:
    """Load simulation_guidelines_agent.md from tau2 data directory."""
    from tau2.utils import DATA_DIR
    guidelines_path = DATA_DIR / "tau2" / "user_simulator" / "simulation_guidelines_agent.md"
    try:
        with open(guidelines_path, "r") as fp:
            return fp.read()
    except FileNotFoundError:
        return (
            "You are an AI agent representing a traveler in a multi-agent booking system.\n"
            "Communicate the traveler's needs clearly and accurately to the platform agent.\n"
            "Keep talking until you receive a reservation confirmation, or until it is clear the booking cannot be completed."
        )

USER_SYSTEM_PROMPT = """{guidelines}

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
        seq_len: int = 25000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.platform_llm = platform_llm
        self.platform_llm_args = platform_llm_args or {}
        self.db_path = db_path
        self.seq_len = seq_len
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
        state["__tau2_max_errors__"] = 10
        state["__tau2_turn_count__"] = 0

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

    async def get_prompt_messages(self, state: State) -> Messages:
        messages = await super().get_prompt_messages(state)
        messages = _strip_thinking(messages)
        state["__full_messages__"] = _msgs_to_dicts(messages)
        return _sliding_window(messages, self.seq_len)

    async def render_completion(self, state: State):
        await super().render_completion(state)
        # Append the final completion to __full_messages__ so reward evaluation sees the complete
        # trace. The stop condition fires after the last model generation without calling
        # get_prompt_messages() again, so the final assistant message would otherwise be missing.
        if "__full_messages__" in state and state.get("trajectory"):
            last_completion = state["trajectory"][-1].get("completion", [])
            state["__full_messages__"] = state["__full_messages__"] + _msgs_to_dicts(last_completion)

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

        # Append the model's latest completion to full history.
        if "__full_messages__" in state:
            state["__full_messages__"] = state["__full_messages__"] + [last_msg]

        # If content is None (reasoning_parser put everything in reasoning_content), use that.
        user_text = last_msg.get("content") or last_msg.get("reasoning_content") or ""
        # Strip <think> blocks before forwarding to platform simulator
        import re as _re
        user_text = _re.sub(r"<think>.*?</think>\s*", "", user_text, flags=_re.DOTALL).strip()

        if not user_text:
            logger.warning("Trained model returned empty/None content — ending rollout")
            state["__tau2_termination__"] = "error"
            state["final_env_response"] = [{"role": "user", "content": "[no response]"}]
            return state["final_env_response"]

        # Check if user said STOP
        if STOP_TOKEN in user_text:
            state["__tau2_termination__"] = "user_stop"
            state["final_env_response"] = [
                {"role": "user", "content": "[Conversation ended by user]"}
            ]
            return state["final_env_response"]

        platform_agent: LLMAgent = state["__tau2_platform_agent__"]
        agent_state: LLMAgentState = state["__tau2_agent_state__"]
        tools: AirlineTools = state["__tau2_tools__"]

        # Forward user message to platform agent
        user_msg = Tau2UserMsg(role="user", content=user_text)

        prev_msg_len = len(agent_state.messages)

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
            state["__tau2_termination__"] = "error"
            state["final_env_response"] = [
                {"role": "user", "content": f"[Platform agent error: {exc}]"}
            ]
            return state["final_env_response"]

        state["__tau2_agent_state__"] = agent_state

        # Accumulate platform agent messages for reward evaluation.
        # These contain tool calls (including book_reservation) that the evaluators need.
        new_msgs = list(agent_state.messages[prev_msg_len:])
        if "__platform_msgs__" not in state:
            state["__platform_msgs__"] = []
        state["__platform_msgs__"].extend(new_msgs)

        # Check for platform agent stop
        if STOP_TOKEN in (agent_response or ""):
            state["__tau2_termination__"] = "agent_stop"
            state["final_env_response"] = [
                {"role": "user", "content": agent_response}
            ]
            return state["final_env_response"]

        # Increment turn count (each text exchange = 1 turn)
        state["__tau2_turn_count__"] = state.get("__tau2_turn_count__", 0) + 1

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
                        from tau2.environment.environment import Environment as Tau2Environment
                        result = Tau2Environment.to_json_str(result)
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


def _msgs_to_dicts(messages: Messages) -> list[dict]:
    """Convert messages (dict or typed objects) to plain dicts."""
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif hasattr(m, "model_dump"):
            result.append(m.model_dump())
        else:
            result.append(dict(m))
    return result


def _sliding_window(messages: Messages, seq_len: int) -> Messages:
    """Truncate messages to fit within seq_len, always keeping system prompt and last message.

    Uses char-count // 4 as a token estimate. Reserves 8192 tokens for model output.
    Only drops messages when the conversation is genuinely too long — most turns are unaffected.

    Drops messages atomically: an assistant message with tool_calls is always dropped together
    with its immediately following tool-result messages, so the window never contains an orphaned
    tool result without its corresponding tool call.
    """
    if len(messages) <= 2:
        return messages

    budget = seq_len - 8192

    def _token_estimate(msgs):
        total = 0
        for m in msgs:
            d = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else dict(m))
            total += (len(str(d.get("content") or "")) + len(str(d.get("tool_calls") or ""))) // 4
        return total

    def _role(m):
        return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")

    def _tool_calls(m):
        return m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)

    def _drop_front_unit(mid: list) -> list:
        """Drop the first atomic unit: an assistant+tool_calls message together with its tool results."""
        if not mid:
            return mid
        i = 1
        if _role(mid[0]) == "assistant" and _tool_calls(mid[0]):
            while i < len(mid) and _role(mid[i]) == "tool":
                i += 1
        return mid[i:]

    if _token_estimate(messages) <= budget:
        return messages

    system = [messages[0]]
    last = [messages[-1]]
    middle = list(messages[1:-1])

    while middle and _token_estimate(system + middle + last) > budget:
        middle = _drop_front_unit(middle)

    if len(middle) < len(messages) - 2:
        logger.warning(
            "Sliding window dropped %d messages to fit within seq_len=%d",
            len(messages) - 2 - len(middle),
            seq_len,
        )

    return system + middle + last


def _strip_thinking(messages: Messages) -> Messages:
    """Remove thinking/reasoning content from assistant messages in context history.

    Think blocks and reasoning_content accumulate across turns and cause context overflow
    with extended-thinking models (e.g. Qwen3 thinking mode). Always returns typed
    AssistantMessage objects (not plain dicts) so verifiers' client can serialize correctly.
    """
    import re
    from verifiers.types import AssistantMessage

    def _clean(content, reasoning_content, tool_calls):
        if content and isinstance(content, str):
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        if not content:
            content = reasoning_content or ""
        return content, (tool_calls if tool_calls is not None else None)

    result = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content, tc = _clean(msg.get("content"), msg.get("reasoning_content"), msg.get("tool_calls"))
            result.append(AssistantMessage(role="assistant", content=content, tool_calls=tc))
        elif hasattr(msg, "role") and msg.role == "assistant":
            rc = getattr(msg, "reasoning_content", None)
            content, tc = _clean(getattr(msg, "content", None), rc, getattr(msg, "tool_calls", None))
            result.append(AssistantMessage(role="assistant", content=content, tool_calls=tc))
        else:
            result.append(msg)
    return result


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


def _prepend_system(prompt: Messages, context: str, system_template: str = USER_SYSTEM_PROMPT) -> Messages:
    """Prepend system message with user context to the prompt."""
    guidelines = _load_user_guidelines()
    system_msg = {
        "role": "system",
        "content": system_template.format(guidelines=guidelines, context=context),
    }
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        return [system_msg] + list(prompt[1:])
    return [system_msg] + list(prompt)


# ---------------------------------------------------------------------------
# Marketplace User Environment
# ---------------------------------------------------------------------------

MARKETPLACE_USER_SYSTEM_PROMPT = """{guidelines}

## Marketplace Mode

You are shopping for the best flight deal across multiple airlines.

You have three tools:
- list_airlines() — see available airlines
- query_airline(airline_id, message) — send a message to a specific airline's customer service
- get_user_details() — get the user's ID and payment methods

IMPORTANT: Airlines can ONLY hear you through query_airline(). Plain text messages are NOT sent to anyone.

Strategy:
1. Call list_airlines() to see options
2. Query multiple airlines to compare prices and availability
3. Negotiate for the best deal matching your user's preferences
4. Book through the airline that offers the best match

{context}"""

MAX_INNER_STEPS = 10  # max tool-call rounds per query_airline call


class Tau2MarketplaceUserEnv(vf.MultiTurnEnv):
    """MultiTurnEnv where the trained model plays the user in a multi-airline marketplace.

    The user agent makes tool calls (list_airlines, query_airline, get_user_details).
    Each airline has its own LLMAgent with carrier-specific policy and filtered tools.
    """

    def __init__(
        self,
        airline_llm: str = "gpt-4.1-mini",
        airline_llm_args: Optional[dict] = None,
        db_path: Optional[str] = None,
        seq_len: int = 25000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.airline_llm = airline_llm
        self.airline_llm_args = airline_llm_args or {}
        self.db_path = db_path
        self.seq_len = seq_len
        self._base_db: Optional[FlightDB] = None

    def _get_base_db(self) -> FlightDB:
        if self._base_db is None:
            self._base_db = load_db(self.db_path)
        return self._base_db

    async def setup_state(self, state: State) -> State:
        """Initialize per-rollout state: fresh DB, airline sessions, user context."""
        info = state["info"]
        if isinstance(info, str):
            info = json.loads(info)
            state["input"]["info"] = info

        task_dict = info.get("task", {})
        gen_meta = task_dict.get("generation_metadata", {})

        # Fresh DB copy per rollout
        db = deepcopy(self._get_base_db())

        # Load base policy
        from tau2.domains.airline_a2a.utils import AIRLINE_A2A_POLICY_PATH
        with open(AIRLINE_A2A_POLICY_PATH, "r") as fp:
            base_policy = fp.read()

        # Build airline sessions from sampled_policies in task metadata
        sampled_policies = gen_meta.get("sampled_policies", {})
        airline_sessions: dict[str, dict] = {}

        if sampled_policies:
            from tau2.domains.airline_a2a.airline_policies import SampledAirlinePolicy, render_policy_prompt, to_seat_meal_config
            for airline_id, raw_policy in sampled_policies.items():
                sampled = SampledAirlinePolicy.model_validate(raw_policy)
                airline_name = getattr(sampled, "airline_name", airline_id)
                # Per-airline tools with carrier filter
                airline_tools = create_airline_tools(
                    deepcopy(db),
                    carrier_filter=sampled.carriers,
                    seat_meal_config=to_seat_meal_config(sampled),
                    exclude_tools=["get_user_details", "run_python", "transfer_to_human_agents"],
                    extra_bag_fee=sampled.baggage.extra_bag_fee,
                )
                apply_initial_state(airline_tools, task_dict)

                # Per-airline policy
                airline_policy = f"{base_policy}\n\n# {airline_name} — Airline-Specific Policy\n\n{render_policy_prompt(sampled)}"

                # Create airline agent
                tau2_tool_defs = list(airline_tools.get_tools().values())
                airline_agent = LLMAgent(
                    tools=tau2_tool_defs,
                    domain_policy=airline_policy,
                    llm=self.airline_llm,
                    llm_args=deepcopy(self.airline_llm_args),
                )

                # Initialize agent with greeting (matches MarketplaceOrchestrator.AirlineSession)
                greeting = Tau2AssistantMsg(
                    role="assistant",
                    content=f"Welcome to {airline_name}! How can I help you?",
                    cost=0.0,
                )
                agent_state = airline_agent.get_init_state(message_history=[greeting])

                airline_sessions[airline_id] = {
                    "name": sampled.airline_name if hasattr(sampled, "airline_name") else airline_id,
                    "agent": airline_agent,
                    "agent_state": agent_state,
                    "tools": airline_tools,
                    "policy": airline_policy,
                }

        # Store environment state
        state["__airline_sessions__"] = airline_sessions
        state["__tau2_error_count__"] = 0
        state["__tau2_max_errors__"] = 10
        state["__tau2_turn_count__"] = 0
        state["__booked_airline__"] = None
        state["__flight_db__"] = db
        state["__platform_msgs__"] = []  # Accumulate all airline agent msgs for evaluation
        state["__marketplace_info__"] = {
            "airlines": {aid: {"name": s["name"], "queries": 0} for aid, s in airline_sessions.items()},
            "airlines_queried": {},
            "booked_airline": None,
        }

        # User gets marketplace tools
        state["tool_defs"] = get_marketplace_tool_defs()

        # Also store task init data for get_user_details tool
        state["__task_dict__"] = task_dict

        # Build user context and system prompt
        user_data = info.get("user_data", {})
        context = _build_user_context(user_data)
        state["input"]["prompt"] = _prepend_system(
            state["prompt"], context, system_template=MARKETPLACE_USER_SYSTEM_PROMPT
        )

        return state

    @vf.stop
    async def too_many_tool_errors(self, state: State) -> bool:
        return state.get("__tau2_error_count__", 0) >= state.get("__tau2_max_errors__", 5)

    async def get_prompt_messages(self, state: State) -> Messages:
        messages = await super().get_prompt_messages(state)
        messages = _strip_thinking(messages)
        state["__full_messages__"] = _msgs_to_dicts(messages)
        return _sliding_window(messages, self.seq_len)

    async def render_completion(self, state: State):
        await super().render_completion(state)
        if "__full_messages__" in state and state.get("trajectory"):
            last_completion = state["trajectory"][-1].get("completion", [])
            state["__full_messages__"] = state["__full_messages__"] + _msgs_to_dicts(last_completion)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """Process the marketplace user's output (tool calls or text)."""
        last_msg = _get_last_assistant(messages)
        if last_msg is None:
            return [{"role": "user", "content": "Welcome to the airline marketplace! Use list_airlines() to get started."}]

        if "__full_messages__" in state:
            state["__full_messages__"] = state["__full_messages__"] + [last_msg]

        tool_calls = last_msg.get("tool_calls")
        content = last_msg.get("content") or ""

        # Check for STOP
        if STOP_TOKEN in content:
            state["__tau2_termination__"] = "user_stop"
            state["final_env_response"] = [{"role": "user", "content": "[Conversation ended by user]"}]
            return state["final_env_response"]

        # Handle tool calls
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("arguments", "{}")
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except json.JSONDecodeError:
                        tc_args = {}

                result = await self._handle_marketplace_tool(tc_name, tc_args, state)
                if result.startswith("Error:"):
                    state["__tau2_error_count__"] = state.get("__tau2_error_count__", 0) + 1
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

            state["__tau2_turn_count__"] = state.get("__tau2_turn_count__", 0) + 1
            return tool_results

        # Plain text — remind user to use tools
        state["__tau2_turn_count__"] = state.get("__tau2_turn_count__", 0) + 1
        return [{
            "role": "user",
            "content": (
                "Your message was NOT sent to any airline. "
                "Airlines can only receive messages through query_airline(). "
                "Use query_airline(airline_id, message) to talk to an airline."
            ),
        }]

    async def _handle_marketplace_tool(
        self, tool_name: str, args: dict, state: State
    ) -> str:
        """Route marketplace tool calls."""
        airline_sessions = state["__airline_sessions__"]

        if tool_name == "list_airlines":
            airlines = [
                {"id": aid, "name": s["name"]}
                for aid, s in airline_sessions.items()
            ]
            return json.dumps(airlines, indent=2)

        elif tool_name == "get_user_details":
            return self._get_user_details(state)

        elif tool_name == "query_airline":
            airline_id = args.get("airline_id", "")
            message = args.get("message", "")
            if airline_id not in airline_sessions:
                available = list(airline_sessions.keys())
                return f"Error: airline '{airline_id}' not found. Available: {available}"
            return await self._query_airline(airline_id, message, state)

        else:
            return f"Error: unknown tool '{tool_name}'. Available: list_airlines, query_airline, get_user_details"

    async def _query_airline(
        self, airline_id: str, message: str, state: State
    ) -> str:
        """Send message to airline agent, run inner loop until text response."""
        session = state["__airline_sessions__"][airline_id]
        agent: LLMAgent = session["agent"]
        agent_state: LLMAgentState = session["agent_state"]
        tools: AirlineTools = session["tools"]

        # Track queries
        mkt_info = state["__marketplace_info__"]
        mkt_info["airlines"][airline_id]["queries"] = mkt_info["airlines"][airline_id].get("queries", 0) + 1
        mkt_info["airlines_queried"][airline_id] = mkt_info["airlines"][airline_id]["queries"]

        user_msg = Tau2UserMsg(role="user", content=message)
        prev_msg_len = len(agent_state.messages)

        try:
            agent_reply, agent_state = await asyncio.to_thread(
                agent.generate_next_message, user_msg, agent_state
            )

            inner_steps = 0
            while agent_reply.tool_calls and inner_steps < MAX_INNER_STEPS:
                tool_results = []
                for tc in agent_reply.tool_calls:
                    try:
                        result = tools.use_tool(tc.name, **tc.arguments)
                        if not isinstance(result, str):
                            result = Tau2Environment.to_json_str(result)
                        tool_results.append(
                            Tau2ToolMsg(id=tc.id, role="tool", content=result)
                        )
                        # Detect successful booking
                        if tc.name == "book_reservation" and "error" not in result.lower()[:50]:
                            state["__booked_airline__"] = airline_id
                            mkt_info["booked_airline"] = airline_id
                    except Exception as exc:
                        state["__tau2_error_count__"] = state.get("__tau2_error_count__", 0) + 1
                        tool_results.append(
                            Tau2ToolMsg(id=tc.id, role="tool", content=f"Error: {exc}", error=True)
                        )

                if len(tool_results) == 1:
                    feed_msg = tool_results[0]
                else:
                    feed_msg = Tau2MultiToolMsg(role="tool", tool_messages=tool_results)

                agent_reply, agent_state = await asyncio.to_thread(
                    agent.generate_next_message, feed_msg, agent_state
                )
                inner_steps += 1

            session["agent_state"] = agent_state

            # Accumulate platform msgs for evaluation
            new_msgs = list(agent_state.messages[prev_msg_len:])
            state["__platform_msgs__"].extend(new_msgs)

            if agent_reply.tool_calls:
                return f"[{session['name']} is still processing after {MAX_INNER_STEPS} tool calls. Try a simpler request.]"

            name = session["name"]
            return f"[{name}]: {agent_reply.content or ''}"

        except Exception as exc:
            logger.warning("Airline %s error: %s", airline_id, exc)
            return f"Error communicating with {session['name']}: {exc}"

    def _get_user_details(self, state: State) -> str:
        """Return user ID and payment methods from task initialization data."""
        task_dict = state.get("__task_dict__", {})
        try:
            init_data = (task_dict.get("initial_state") or {}).get("initialization_data")
            agent_data = (init_data.get("agent_data") if init_data else None) or {}
            users = agent_data.get("users", {})
            if not users:
                return json.dumps({"user_id": None, "payment_methods": []})

            user_id, user_data = next(iter(users.items()))
            raw_payments = user_data.get("payment_methods", {})
            payment_methods = []
            for pm_id, pm in raw_payments.items():
                entry = {"id": pm_id, "source": pm.get("source", "unknown")}
                if pm.get("source") == "credit_card":
                    entry["brand"] = pm.get("brand", "")
                    entry["last_four"] = pm.get("last_four", "")
                elif pm.get("source") in ("gift_card", "certificate"):
                    entry["amount"] = pm.get("amount", 0)
                payment_methods.append(entry)

            return json.dumps({"user_id": user_id, "payment_methods": payment_methods}, indent=2)
        except Exception as e:
            logger.warning("get_user_details failed: %s", e)
            return f"Error retrieving user details: {e}"
