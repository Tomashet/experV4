# scripts/common.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import highway_env  # registers merge-v0 / highway-v0

from src.context import MarkovContextScheduler
from src.safety import SafetyParams, ConformalCalibrator
from src.wrappers import (
    ContextNonstationaryWrapper,
    ObservationNoiseWrapper,
    SafetyShieldWrapper,
    FixedKinematicsObsWrapper,
)


def _normalize_env_id(env_id: str) -> str:
    env_id = (env_id or "").strip()
    if env_id == "merge":
        return "merge-v0"
    if env_id == "highway":
        return "highway-v0"
    return env_id


def make_env(
    env_id: str,
    seed: int,
    action_space_type: str,   # "discrete" or "continuous"
    p_stay: float,
    no_mpc: bool,
    no_conformal: bool,
    safety_params: SafetyParams,
) -> Tuple[gym.Env, Optional[Any], Optional[Any]]:
    env_id = _normalize_env_id(env_id)

    # 1) Base env
    env = gym.make(env_id, render_mode=None, disable_env_checker=True)

    # --- CRITICAL FIX FOR SAC: set continuous action space BEFORE SB3 builds the model ---
    if action_space_type.lower() == "continuous":
        # highway-env supports configuring action type via env.unwrapped.configure
        if hasattr(env.unwrapped, "configure"):
            env.unwrapped.configure(
                {
                    "action": {"type": "ContinuousAction"}
                }
            )
        else:
            raise RuntimeError("Env does not support configure(); cannot set ContinuousAction for SAC.")

    # Seed base env deterministically
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    # 2) Markov context switching
    scheduler = MarkovContextScheduler(seed=seed, p_stay=p_stay)
    env = ContextNonstationaryWrapper(env, scheduler=scheduler)

    # 3) Context-dependent observation noise/dropout
    env = ObservationNoiseWrapper(env, seed=seed)

    # 4) Optional conformal calibrator
    calibrator: Optional[ConformalCalibrator] = None
    if not no_conformal:
        calibrator = ConformalCalibrator(params=safety_params)

    # 5) Safety shield wrapper
    env = SafetyShieldWrapper(
        env,
        params=safety_params,
        action_space_type=action_space_type,
        no_mpc=no_mpc,
        no_conformal=no_conformal,
        calibrator=calibrator,
    )

    # 6) Fix observation tensor shape for SB3 buffers
    env = FixedKinematicsObsWrapper(env, K=10)

    return env, None, calibrator