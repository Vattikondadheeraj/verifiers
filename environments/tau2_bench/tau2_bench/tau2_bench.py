"""Entry point for the tau2-bench environment integration with verifiers.

Usage:
    env = load_environment(role="platform")  # Train platform agent
    env = load_environment(role="user")      # Train user agent
    env = load_environment(role="both")      # EnvGroup with both
"""

from __future__ import annotations

from typing import Optional

import verifiers as vf

from .tau2_data import build_platform_dataset, build_user_dataset
from .tau2_platform_env import Tau2PlatformEnv
from .tau2_rewards import build_rubric
from .tau2_user_env import Tau2UserEnv


def load_environment(
    role: str = "both",
    user_llm: str = "gpt-4.1-mini",
    user_llm_args: Optional[dict] = None,
    platform_llm: str = "gpt-4.1-mini",
    platform_llm_args: Optional[dict] = None,
    max_turns: int = 20,
    num_tasks: int = -1,
    task_split_name: Optional[str] = "base",
    db_path: Optional[str] = None,
    email_prefs_path: Optional[str] = None,
    preference_profiles_path: Optional[str] = None,
    platform_reward_basis: Optional[str] = None,
    user_reward_basis: Optional[str] = None,
    sidecar_dir: Optional[str] = None,
    **kwargs,
) -> vf.Environment:
    """Load the tau2-bench environment.

    Args:
        role: Which agent to train — ``"platform"``, ``"user"``, or ``"both"``.
        user_llm: LLM for the simulated user (platform env) or user agent (user env).
        user_llm_args: Extra kwargs for the user LLM.
        platform_llm: LLM for the simulated platform agent (user env).
        platform_llm_args: Extra kwargs for the platform LLM.
        max_turns: Maximum conversation turns per rollout.
        num_tasks: Number of tasks to use (-1 = all).
        task_split_name: Task split name (default "base").
        db_path: Override path to the airline database JSON.
        email_prefs_path: Override path to user email preferences.
        preference_profiles_path: Override path to user preference profiles.
        **kwargs: Extra arguments forwarded to the environment constructors.
    """
    platform_rb = [r.strip() for r in platform_reward_basis.split(",")] if platform_reward_basis else None
    user_rb = [r.strip() for r in user_reward_basis.split(",")] if user_reward_basis else None
    parser = vf.Parser()

    envs = {}

    if role in ("platform", "both"):
        platform_dataset = build_platform_dataset(
            num_tasks=num_tasks,
            task_split_name=task_split_name,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
        )
        platform_env = Tau2PlatformEnv(
            dataset=platform_dataset,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            db_path=db_path,
            max_turns=max_turns,
            parser=parser,
            rubric=build_rubric(reward_basis=platform_rb, sidecar_dir=sidecar_dir),
            **kwargs,
        )
        envs["platform"] = platform_env

    if role in ("user", "both"):
        user_dataset = build_user_dataset(
            num_tasks=num_tasks,
            task_split_name=task_split_name,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
        )
        user_env = Tau2UserEnv(
            dataset=user_dataset,
            platform_llm=platform_llm,
            platform_llm_args=platform_llm_args,
            db_path=db_path,
            max_turns=max_turns,
            parser=parser,
            rubric=build_rubric(reward_basis=user_rb, sidecar_dir=sidecar_dir),
            **kwargs,
        )
        envs["user"] = user_env

    if role == "both":
        return vf.EnvGroup(
            envs=list(envs.values()),
            env_names=list(envs.keys()),
        )
    elif role == "platform":
        return envs["platform"]
    elif role == "user":
        return envs["user"]
    else:
        raise ValueError(f"Unknown role: {role!r}. Must be 'platform', 'user', or 'both'.")
