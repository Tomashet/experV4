# src/wrappers.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym

from src.safety import SafetyParams, ConformalCalibrator, clearance_margin


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _pad_trunc_2d(a: np.ndarray, K: int) -> np.ndarray:
    """
    Pads/truncates a 2D array to K rows, preserving columns.
    Returns shape (K, D). IMPORTANT: do NOT flatten for merge-v0.
    """
    if a.ndim != 2:
        return a
    n, d = a.shape
    if n == K:
        return a
    if n > K:
        return a[:K]
    pad = np.zeros((K - n, d), dtype=a.dtype)
    return np.concatenate([a, pad], axis=0)


def _fix_obs(obs: Any, K: int) -> Any:
    """
    Makes observation shape stable:
      - If obs is 2D (N x D): pad/truncate to (K x D) and keep 2D.
      - If obs is 1D: keep as-is.
      - If obs is dict: apply to key 'observation' if present, else first array-like.
    """
    if isinstance(obs, dict):
        out = dict(obs)
        if "observation" in out:
            out["observation"] = _fix_obs(out["observation"], K)
            return out
        for k, v in out.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                out[k] = _fix_obs(v, K)
                break
        return out

    a = _to_np(obs).astype(np.float32, copy=False)
    if a.ndim == 2:
        return _pad_trunc_2d(a, K)
    return a


class ContextNonstationaryWrapper(gym.Wrapper):
    """
    Maintains a latent context id driven by a scheduler (e.g., MarkovContextScheduler).
    Exposes: self.cur_ctx_id (int)
    Adds to info: info['ctx_id']
    """

    def __init__(self, env: gym.Env, scheduler: Any):
        super().__init__(env)
        self.scheduler = scheduler
        self.cur_ctx_id: int = 0

    def _scheduler_reset(self) -> int:
        if hasattr(self.scheduler, "reset"):
            v = self.scheduler.reset()
            if v is None and hasattr(self.scheduler, "cur_ctx_id"):
                v = getattr(self.scheduler, "cur_ctx_id")
            if v is not None:
                return int(v)
        return 0

    def _scheduler_step(self) -> int:
        if hasattr(self.scheduler, "step"):
            v = self.scheduler.step()
            if v is None and hasattr(self.scheduler, "cur_ctx_id"):
                v = getattr(self.scheduler, "cur_ctx_id")
            if v is not None:
                return int(v)
        return self.cur_ctx_id

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cur_ctx_id = self._scheduler_reset()
        if isinstance(info, dict):
            info = dict(info)
            info["ctx_id"] = self.cur_ctx_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cur_ctx_id = self._scheduler_step()
        if isinstance(info, dict):
            info = dict(info)
            info["ctx_id"] = self.cur_ctx_id
        return obs, reward, terminated, truncated, info


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds mild Gaussian noise to observations.
    If env has cur_ctx_id, noise can vary with context id.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
        base_sigma: float = 0.0,
        ctx_sigma: float = 0.01,
    ):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.base_sigma = float(base_sigma)
        self.ctx_sigma = float(ctx_sigma)

    def observation(self, obs):
        ctx = getattr(self.env, "cur_ctx_id", 0)
        try:
            ctx = int(ctx)
        except Exception:
            ctx = 0

        sigma = self.base_sigma + self.ctx_sigma * max(0, ctx)
        if sigma <= 0:
            return obs

        if isinstance(obs, dict):
            out = dict(obs)
            if "observation" in out:
                a = _to_np(out["observation"]).astype(np.float32, copy=False)
                out["observation"] = a + self.rng.normal(0.0, sigma, size=a.shape).astype(np.float32)
                return out
            for k, v in out.items():
                if isinstance(v, (np.ndarray, list, tuple)):
                    a = _to_np(v).astype(np.float32, copy=False)
                    out[k] = a + self.rng.normal(0.0, sigma, size=a.shape).astype(np.float32)
                    break
            return out

        a = _to_np(obs).astype(np.float32, copy=False)
        return a + self.rng.normal(0.0, sigma, size=a.shape).astype(np.float32)


class FixedKinematicsObsWrapper(gym.ObservationWrapper):
    """
    Ensures stable observation shape for SB3 VecEnv buffers.

    For merge-v0 your observation space is (5,5). SB3 expects that shape,
    so we keep 2D observations and pad/truncate to (K, D).
    """

    def __init__(self, env: gym.Env, K: int = 5):
        super().__init__(env)
        self.K = int(K)

        # If underlying obs space is 2D Box, update it to (K, D)
        try:
            if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 2:
                _, d = env.observation_space.shape
                low = np.full((self.K, d), np.min(env.observation_space.low), dtype=np.float32)
                high = np.full((self.K, d), np.max(env.observation_space.high), dtype=np.float32)
                self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        except Exception:
            # If we fail to set, SB3 might still work if shapes match anyway.
            pass

    def observation(self, obs):
        return _fix_obs(obs, self.K)


class SafetyShieldWrapper(gym.Wrapper):
    """
    Filters actions through the MPC-like safety shield defined in src/safety.py.
    Uses MPCLikeSafetyShield when available.

    Adds shield diagnostics to info as:
      info['shield/<key>'] = value
    """

    def __init__(
        self,
        env: gym.Env,
        params: SafetyParams,
        action_space_type: str,  # "discrete"|"continuous"
        no_mpc: bool = False,
        no_conformal: bool = False,
        calibrator: Optional[ConformalCalibrator] = None,
    ):
        super().__init__(env)
        self.params = params
        self.action_space_type = str(action_space_type)
        self.no_mpc = bool(no_mpc)
        self.no_conformal = bool(no_conformal)
        self.calibrator = calibrator

        import src.safety as s

        candidates = [
            "MPCLikeSafetyShield",  # <-- actual class in your src/safety.py
            "SafetyShield",
            "Shield",
            "SafetyLayer",
            "MPCShield",
        ]

        shield_cls = None
        chosen = None
        for name in candidates:
            if hasattr(s, name):
                shield_cls = getattr(s, name)
                chosen = name
                break

        if shield_cls is None:
            raise ImportError(
                "Could not find a shield class in src/safety.py. Tried: "
                + ", ".join(candidates)
                + '. Run: python -c "import src.safety as s; print([n for n in dir(s) if \'hield\' in n.lower()])" '
                "and then update SafetyShieldWrapper to use the correct name."
            )

        # Instantiate with the MPCLikeSafetyShield signature
        if chosen == "MPCLikeSafetyShield":
            self.shield = shield_cls(
                params=self.params,
                action_space_type=self.action_space_type,
                no_mpc=self.no_mpc,
                no_conformal=self.no_conformal,
                calibrator=self.calibrator,
            )
        else:
            # Best-effort legacy instantiation
            try:
                self.shield = shield_cls(
                    params=self.params,
                    action_space_type=self.action_space_type,
                    no_mpc=self.no_mpc,
                    no_conformal=self.no_conformal,
                    calibrator=self.calibrator,
                )
            except TypeError:
                self.shield = shield_cls(self.params)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        cur_ctx_id = getattr(self.env, "cur_ctx_id", 0)
        try:
            cur_ctx_id = int(cur_ctx_id)
        except Exception:
            cur_ctx_id = 0

        safe_action, shield_info = self.shield.filter_action(self.env, action, cur_ctx_id=cur_ctx_id)

        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # Update calibrator with a simple, robust clearance shortfall signal
        if (self.calibrator is not None) and (not self.no_conformal):
            try:
                d = clearance_margin(self.env, self.params)
                if np.isfinite(d):
                    err = max(0.0, float(self.params.epsilon - d))
                    self.calibrator.update(err)
            except Exception:
                pass

        if isinstance(info, dict):
            info = dict(info)
            if isinstance(shield_info, dict):
                for k, v in shield_info.items():
                    info[f"shield/{k}"] = v
            # simple flag
            if self.action_space_type == "discrete":
                info["shield/filtered"] = (safe_action != action)
            else:
                info["shield/filtered"] = (safe_action is not action)

        return obs, reward, terminated, truncated, info