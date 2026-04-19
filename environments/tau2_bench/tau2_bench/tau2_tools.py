"""Convert tau2 AirlineTools into verifiers-compatible tool definitions.

Supports:
  - Platform agent tools (search_direct_flight, book_reservation, etc.)
  - Marketplace user tools (list_airlines, query_airline, get_user_details)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from tau2.domains.airline_a2a.data_model import FlightDB
from tau2.domains.airline_a2a.tools import AirlineTools
from tau2.environment.environment import Environment as Tau2Environment
from tau2.environment.tool import Tool as Tau2Tool

from verifiers.types import Tool as VFTool

logger = logging.getLogger(__name__)

# Tools to exclude from the platform agent's tool list.
# run_python uses signal.alarm which crashes in async threads.
PLATFORM_EXCLUDED_TOOLS = {"run_python"}


# ---------------------------------------------------------------------------
# Platform agent tools (single-airline)
# ---------------------------------------------------------------------------


def create_airline_tools(
    db: FlightDB,
    carrier_filter: Optional[list[str]] = None,
    seat_meal_config: Optional[dict] = None,
    exclude_tools: Optional[list[str]] = None,
    extra_bag_fee: int = 50,
) -> AirlineTools:
    """Create an AirlineTools instance, optionally with carrier filtering for marketplace."""
    return AirlineTools(
        db,
        carrier_filter=carrier_filter,
        seat_meal_config=seat_meal_config,
        exclude_tools=exclude_tools,
        extra_bag_fee=extra_bag_fee,
    )


def get_vf_tool_defs(tools: AirlineTools) -> list[VFTool]:
    """Convert tau2 AirlineTools to verifiers Tool definitions."""
    tau2_tools: dict[str, Tau2Tool] = tools.get_tools()
    vf_tools: list[VFTool] = []
    for _name, tau2_tool in tau2_tools.items():
        schema = tau2_tool.openai_schema  # {"type": "function", "function": {...}}
        func_schema = schema["function"]
        if func_schema["name"] in PLATFORM_EXCLUDED_TOOLS:
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
    """Execute a tool call against AirlineTools, returning the result as a string."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {arguments}"

    if not tools.has_tool(tool_name):
        return f"Error: unknown tool '{tool_name}'"

    try:
        result = await asyncio.to_thread(tools.use_tool, tool_name, **arguments)
        return Tau2Environment.to_json_str(result)
    except Exception as exc:
        logger.warning("Tool call %s failed: %s", tool_name, exc)
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Marketplace user tools (list_airlines, query_airline, get_user_details)
# ---------------------------------------------------------------------------


MARKETPLACE_TOOL_DEFS: list[VFTool] = [
    VFTool(
        name="list_airlines",
        description="List all available airlines in the marketplace.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    VFTool(
        name="query_airline",
        description=(
            "Send a message to a specific airline's customer service agent and get their response. "
            "This is the ONLY way to communicate with airlines."
        ),
        parameters={
            "type": "object",
            "properties": {
                "airline_id": {
                    "type": "string",
                    "description": "The airline identifier (e.g. 'skyhigh', 'budgetwings', 'ultravalue').",
                },
                "message": {
                    "type": "string",
                    "description": "Your message to the airline's customer service agent.",
                },
            },
            "required": ["airline_id", "message"],
        },
    ),
    VFTool(
        name="get_user_details",
        description="Get the user's ID and available payment methods.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


def get_marketplace_tool_defs() -> list[VFTool]:
    """Return the tool definitions for the marketplace user agent."""
    return list(MARKETPLACE_TOOL_DEFS)


# ---------------------------------------------------------------------------
# DB loading and initialization
# ---------------------------------------------------------------------------


def load_db(db_path: str | None = None) -> FlightDB:
    """Load FlightDB from the default or provided path."""
    if db_path:
        return FlightDB.load(db_path)
    from tau2.domains.airline_a2a.utils import AIRLINE_A2A_DB_PATH
    return FlightDB.load(AIRLINE_A2A_DB_PATH)


def apply_initial_state(tools: AirlineTools, task_dict: dict) -> None:
    """Apply a task's initial_state to the AirlineTools instance."""
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
                    logger.warning("Initial action %s failed: %s", func_name, exc)
