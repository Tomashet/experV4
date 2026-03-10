# scripts/common.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401 (register envs)

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

    # 2) Seeding
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    # 3) Markov context switching
    scheduler = MarkovContextScheduler(seed=seed, p_stay=p_stay)
    env = ContextNonstationaryWrapper(env, scheduler=scheduler)

    # 4) Optional observation noise
    env = ObservationNoiseWrapper(env, seed=seed)

    # 5) Optional conformal calibrator (FIXED)
    calibrator: Optional[ConformalCalibrator] = None
    if not no_conformal:
        calibrator = ConformalCalibrator(alpha=0.1, window=200, seed=seed)

    # 6) Safety shield wrapper
    env = SafetyShieldWrapper(
        env,
        params=safety_params,
        action_space_type=action_space_type,
        no_mpc=no_mpc,
        no_conformal=no_conformal,
        calibrator=calibrator,
    )

    # 7) IMPORTANT: keep merge-v0 obs 2D (5x5) so SB3 VecEnv buffers match.
    # Use K=5 for merge-v0; highway-v0 can be different, but this is safe here.
    if env_id == "merge-v0":
        env = FixedKinematicsObsWrapper(env, K=5)
    else:
        # For highway-v0, if obs is 1D this wrapper will leave it alone.
        env = FixedKinematicsObsWrapper(env, K=5)

    return env, None, calibrator