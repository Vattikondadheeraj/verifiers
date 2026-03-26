"""Convert tau2 AirlineTools into verifiers-compatible tool definitions."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any

from tau2.domains.airline_a2a.data_model import FlightDB
from tau2.domains.airline_a2a.tools import AirlineTools
from tau2.environment.environment import Environment as Tau2Environment
from tau2.environment.tool import Tool as Tau2Tool

from verifiers.types import Tool as VFTool

logger = logging.getLogger(__name__)


def create_airline_tools(db: FlightDB) -> AirlineTools:
    """Create a fresh AirlineTools instance from a FlightDB."""
    return AirlineTools(db)


def get_vf_tool_defs(tools: AirlineTools) -> list[VFTool]:
    """Convert tau2 AirlineTools to verifiers Tool definitions.

    tau2's ``get_tools()`` returns ``Dict[str, tau2.environment.tool.Tool]``
    objects with an ``openai_schema`` property that produces the OpenAI
    function-calling format.  We extract the relevant fields and build
    provider-agnostic ``verifiers.types.Tool`` objects.
    """
    # Tools to exclude from the model's tool list.
    # run_python uses signal.alarm which crashes in async threads.
    EXCLUDED_TOOLS = {"run_python"}

    tau2_tools: dict[str, Tau2Tool] = tools.get_tools()
    vf_tools: list[VFTool] = []
    for _name, tau2_tool in tau2_tools.items():
        schema = tau2_tool.openai_schema  # {"type": "function", "function": {...}}
        func_schema = schema["function"]
        if func_schema["name"] in EXCLUDED_TOOLS:
            continue
        vf_tools.append(
            VFTool(
                name=func_schema["name"],
                description=func_schema.get("description", ""),
                parameters=func_schema.get("parameters", {}),
            )
        )
    return vf_tools


async def execute_tool_call(
    tools: AirlineTools,
    tool_name: str,
    arguments: dict[str, Any] | str,
) -> str:
    """Execute a tool call against AirlineTools, returning the result as a string.

    Runs the synchronous tool method in a thread to avoid blocking the
    async event loop.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {arguments}"

    if not tools.has_tool(tool_name):
        return f"Error: unknown tool '{tool_name}'"

    try:
        result = await asyncio.to_thread(tools.use_tool, tool_name, **arguments)
        # Use tau2's own serialization to match what evaluate_simulation expects
        return Tau2Environment.to_json_str(result)
    except Exception as exc:
        logger.warning("Tool call %s failed: %s", tool_name, exc)
        return f"Error: {exc}"


def load_db(db_path: str | None = None) -> FlightDB:
    """Load FlightDB from the default or provided path."""
    if db_path:
        return FlightDB.load(db_path)
    from tau2.domains.airline_a2a.utils import AIRLINE_A2A_DB_PATH

    return FlightDB.load(AIRLINE_A2A_DB_PATH)


def apply_initial_state(tools: AirlineTools, task_dict: dict) -> None:
    """Apply a task's initial_state to the AirlineTools instance.

    Handles both ``initialization_data`` (DB patches) and
    ``initialization_actions`` (function calls on the environment).
    """
    initial_state = task_dict.get("initial_state")
    if not initial_state:
        return

    init_data = initial_state.get("initialization_data")
    if init_data:
        agent_data = init_data.get("agent_data")
        if agent_data:
            tools.update_db(agent_data)

    init_actions = initial_state.get("initialization_actions")
    if init_actions:
        for action in init_actions:
            env_type = action.get("env_type", "assistant")
            if env_type == "assistant":
                func_name = action["func_name"]
                arguments = action.get("arguments", {})
                try:
                    tools.use_tool(func_name, **arguments)
                except Exception as exc:
                    logger.warning(
                        "Initial action %s failed: %s", func_name, exc
                    )
