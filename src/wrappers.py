# src/wrappers.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym

from src.context import MarkovContextScheduler, context_to_highway_config
from src.safety import SafetyParams, ConformalCalibrator


class ContextNonstationaryWrapper(gym.Wrapper):
    """
    Applies a Markov context switch once per episode and configures the underlying highway-env.
    Stores last_config for downstream wrappers (noise/dropout, logging).

    IMPORTANT: attaches ctx_id/ctx_tuple to both reset() AND step() info so step-level
    logging (SB3 callback) can visualize context switching.
    """

    def __init__(self, env: gym.Env, scheduler: MarkovContextScheduler):
        super().__init__(env)
        self.scheduler = scheduler
        self.last_config: Dict[str, Any] = {}
        self._first_reset = True

        # Current context metadata
        self._ctx_id: int = -1
        self._ctx_tuple: Any = None

    def reset(self, **kwargs):
        # Advance context once per new episode (not on the very first reset)
        if self._first_reset:
            ctx = self.scheduler.current()
            self._first_reset = False
        else:
            ctx = self.scheduler.step_episode()

        cfg = context_to_highway_config(ctx)
        self.last_config = cfg
        self._ctx_id = int(cfg.get("_ctx_id", -1))
        self._ctx_tuple = cfg.get("_ctx_tuple", None)

        # Configure underlying env if possible
        if hasattr(self.env.unwrapped, "configure"):
            self.env.unwrapped.configure(cfg)

        obs, info = self.env.reset(**kwargs)

        # Ensure ctx metadata is visible in info (reset)
        info = dict(info) if info is not None else {}
        info["ctx_id"] = self._ctx_id
        info["ctx_tuple"] = self._ctx_tuple
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure ctx metadata is visible in info (every step)
        info = dict(info) if info is not None else {}
        info["ctx_id"] = self._ctx_id
        info["ctx_tuple"] = self._ctx_tuple
        return obs, reward, terminated, truncated, info


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise and/or dropout to observations based on context config keys:
      - _ctx_obs_noise_std
      - _ctx_dropout_prob

    This wrapper expects the wrapped env (or another wrapper) to expose .last_config
    (ContextNonstationaryWrapper provides it).
    """

    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def observation(self, observation):
        obs = np.asarray(observation).astype(np.float32, copy=False)

        cfg = getattr(self.env, "last_config", {}) or {}
        noise_std = float(cfg.get("_ctx_obs_noise_std", 0.0))
        dropout_prob = float(cfg.get("_ctx_dropout_prob", 0.0))

        if noise_std > 0:
            obs = obs + self.rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)

        if dropout_prob > 0:
            mask = self.rng.random(obs.shape) < dropout_prob
            obs = obs.copy()
            obs[mask] = 0.0

        return obs


class SafetyShieldWrapper(gym.Wrapper):
    """
    Wraps an env with an (optional) safety shield.

    IMPORTANT: We deliberately do NOT import a specific 'SafetyShield' class at module import
    time, because your src/safety.py may use a different class name. Instead, we lazy-import
    and search for a shield class only when needed.

    If --no_mpc and --no_conformal are both True, this wrapper becomes a pass-through that
    still logs shield_used/shield_reason fields.

    Also defines a clear safety metric:
      - violation := crashed/collision from highway-env info (if provided)
    """

    def __init__(
        self,
        env: gym.Env,
        params: SafetyParams,
        action_space_type: str,
        no_mpc: bool,
        no_conformal: bool,
        calibrator: Optional[ConformalCalibrator] = None,
    ):
        super().__init__(env)
        self.params = params
        self.action_space_type = action_space_type
        self.no_mpc = bool(no_mpc)
        self.no_conformal = bool(no_conformal)
        self.calibrator = calibrator

        self.shield = None

        # Only try to build a shield if at least one feature is enabled
        if not (self.no_mpc and self.no_conformal):
            from src import safety as safety_mod

            # Try common class names (pick first that exists)
            ShieldCls = (
                getattr(safety_mod, "SafetyShield", None)
                or getattr(safety_mod, "Shield", None)
                or getattr(safety_mod, "SafetyLayer", None)
                or getattr(safety_mod, "MPCShield", None)
            )

            if ShieldCls is None:
                raise ImportError(
                    "Could not find a shield class in src/safety.py. "
                    "Tried: SafetyShield, Shield, SafetyLayer, MPCShield. "
                    "Run: python -c \"import src.safety as s; print([n for n in dir(s) if 'hield' in n.lower()])\" "
                    "and then update SafetyShieldWrapper to use the correct name."
                )

            try:
                self.shield = ShieldCls(
                    params=params,
                    action_space_type=action_space_type,
                    no_mpc=no_mpc,
                    no_conformal=no_conformal,
                    calibrator=calibrator,
                )
            except TypeError:
                # Fallback: try positional args
                self.shield = ShieldCls(params, action_space_type, no_mpc, no_conformal, calibrator)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        shield_used = False
        shield_reason = ""
        proposed_action = action

        if self.shield is not None:
            try:
                # shield.filter_action(env, action) -> (safe_action, used, reason)
                proposed_action, shield_used, shield_reason = self.shield.filter_action(self.env, action)  # type: ignore[attr-defined]
            except Exception:
                try:
                    # shield(action=..., env=...) -> dict with "action"/"shield_used"/"shield_reason"
                    out = self.shield(action=action, env=self.env)  # type: ignore[misc]
                    proposed_action = out.get("action", action)
                    shield_used = bool(out.get("shield_used", False))
                    shield_reason = str(out.get("shield_reason", ""))
                except Exception:
                    proposed_action = action
                    shield_used = False
                    shield_reason = "shield_error_fallback"

        obs, reward, terminated, truncated, info = self.env.step(proposed_action)

        info = dict(info) if info is not None else {}
        info["shield_used"] = bool(shield_used)
        info["shield_reason"] = shield_reason

        # --- Define safety violation signal ---
        # highway-env commonly uses "crashed" (boolean) to indicate a collision.
        crashed = bool(info.get("crashed", False) or info.get("collision", False))
        info["violation"] = crashed

        # Keep near_miss key stable for logging/plots; define later if desired.
        info.setdefault("near_miss", False)

        return obs, reward, terminated, truncated, info


class FixedKinematicsObsWrapper(gym.ObservationWrapper):
    """
    Force a fixed (K, F) observation by truncating/padding on the first axis.

    This solves SB3 buffer mismatch when highway-env returns different (N, F)
    depending on config/context (e.g., merge defaults to 5 vehicles but contexts use 10).
    """

    def __init__(self, env: gym.Env, K: int = 10):
        super().__init__(env)
        self.K = int(K)

        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 2:
            raise TypeError(f"Expected Box((N,F)) obs space, got {space}")

        _, F = space.shape
        self.F = int(F)

        low = np.full((self.K, self.F), -np.inf, dtype=space.dtype)
        high = np.full((self.K, self.F), np.inf, dtype=space.dtype)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(self.K, self.F), dtype=space.dtype
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim != 2:
            return obs

        N, F = obs.shape
        out = np.zeros((self.K, F), dtype=np.float32)

        ncopy = min(N, self.K)
        out[:ncopy] = obs[:ncopy].astype(np.float32, copy=False)
        return out