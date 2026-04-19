"""Entry point for the tau2-bench environment integration with verifiers.

Usage:
    env = load_environment(role="platform")      # Train platform agent (single-airline)
    env = load_environment(role="user")          # Train user agent (single-airline)
    env = load_environment(role="marketplace")   # Train user agent (multi-airline marketplace)
    env = load_environment(role="both")          # EnvGroup with platform + user
"""

from __future__ import annotations

from typing import Optional

import verifiers as vf

from .tau2_data import build_marketplace_dataset, build_platform_dataset, build_user_dataset
from .tau2_platform_env import Tau2PlatformEnv
from .tau2_rewards import build_rubric
from .tau2_user_env import Tau2MarketplaceUserEnv, Tau2UserEnv


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
    seq_len: int = 25000,
    email_prefs_path: Optional[str] = None,
    preference_profiles_path: Optional[str] = None,
    platform_reward_basis: Optional[str] = None,
    user_reward_basis: Optional[str] = None,
    sidecar_dir: Optional[str] = None,
    enable_leakage: bool = False,
    disclosure_level: str = "natural",
    carrier_id: Optional[str] = None,
    **kwargs,
) -> vf.Environment:
    """Load the tau2-bench environment.

    Args:
        role: Which agent to train:
            ``"platform"`` — single-airline customer service agent
            ``"user"`` — single-airline user agent (text-based)
            ``"marketplace"`` — multi-airline user agent (tool-based)
            ``"both"`` — EnvGroup with platform + user
        user_llm: LLM for the simulated user (platform env) or counterpart.
        user_llm_args: Extra kwargs for the user LLM.
        platform_llm: LLM for the simulated platform agent (user env).
        platform_llm_args: Extra kwargs for the platform LLM.
        max_turns: Maximum conversation turns per rollout.
        num_tasks: Number of tasks to use (-1 = all).
        task_split_name: Task split name (default "base").
        db_path: Override path to the airline database JSON.
        seq_len: Max sequence length for sliding window.
        email_prefs_path: Override path to user email preferences.
        preference_profiles_path: Override path to user preference profiles.
        platform_reward_basis: Comma-separated reward types for platform, e.g.
            "BOOKING_ACCURACY,POLICY_COMPLIANCE,PLATFORM_REVENUE".
        user_reward_basis: Comma-separated reward types for user, e.g.
            "PREFERENCE_SATISFACTION" or "PREFERENCE_SATISFACTION,DATA_LEAKAGE".
        sidecar_dir: If set, write per-rollout JSON sidecars for debugging.
        enable_leakage: Enable DATA_LEAKAGE evaluator (requires LLM judge, slow).
        disclosure_level: User disclosure level: "full", "partial", "minimal", "natural".
        carrier_id: For platform training with a specific airline's policy.
        **kwargs: Extra arguments forwarded to environment constructors.
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
            seq_len=seq_len,
            carrier_id=carrier_id,
            max_turns=max_turns,
            parser=parser,
            rubric=build_rubric(
                reward_basis=platform_rb,
                sidecar_dir=sidecar_dir,
                enable_leakage=False,  # Leakage is user-side only
            ),
            **kwargs,
        )
        envs["platform"] = platform_env

    if role in ("user", "both"):
        user_dataset = build_user_dataset(
            num_tasks=num_tasks,
            task_split_name=task_split_name,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
            disclosure_level=disclosure_level,
        )
        user_env = Tau2UserEnv(
            dataset=user_dataset,
            platform_llm=platform_llm,
            platform_llm_args=platform_llm_args,
            db_path=db_path,
            seq_len=seq_len,
            max_turns=max_turns,
            parser=parser,
            rubric=build_rubric(
                reward_basis=user_rb,
                sidecar_dir=sidecar_dir,
                enable_leakage=enable_leakage,
            ),
            **kwargs,
        )
        envs["user"] = user_env

    if role == "marketplace":
        marketplace_dataset = build_marketplace_dataset(
            num_tasks=num_tasks,
            task_split_name=task_split_name,
            email_prefs_path=email_prefs_path,
            preference_profiles_path=preference_profiles_path,
            disclosure_level=disclosure_level,
        )
        marketplace_env = Tau2MarketplaceUserEnv(
            dataset=marketplace_dataset,
            airline_llm=platform_llm,
            airline_llm_args=platform_llm_args,
            db_path=db_path,
            seq_len=seq_len,
            max_turns=max_turns,
            parser=parser,
            rubric=build_rubric(
                reward_basis=user_rb,
                sidecar_dir=sidecar_dir,
                enable_leakage=enable_leakage,
            ),
            **kwargs,
        )
        return marketplace_env

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
        raise ValueError(
            f"Unknown role: {role!r}. Must be 'platform', 'user', 'marketplace', or 'both'."
        )
