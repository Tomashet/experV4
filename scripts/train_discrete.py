# scripts/train_discrete.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure

from src.safety import SafetyParams
from src.logging_utils import ensure_dir, save_json, append_csv
from .common import make_env

# If your repo has scripts/presets.py with get_preset(), keep this import.
# If not, replace with your preset loader.
try:
    from .presets import get_preset
except Exception:
    get_preset = None


def _normalize_env_id(env: str) -> str:
    """Accepts merge/merge-v0/highway/highway-v0 and returns a valid gym id."""
    if env is None:
        return "highway-v0"
    e = str(env).strip()
    # common mistakes / shorthand
    if e == "merge":
        return "merge-v0"
    if e == "highway":
        return "highway-v0"
    return e


class TrainLoggerCallback(BaseCallback):
    def __init__(self, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = os.path.join(run_dir, "train_monitor.csv")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[-1]
            append_csv(
                self.csv_path,
                {
                    "timestep": int(self.num_timesteps),
                    "clearance": info.get("clearance", np.nan),
                    "violation": int(bool(info.get("violation", False))),
                    "near_miss": int(bool(info.get("near_miss", False))),
                    "shield_used": int(bool(info.get("shield_used", False))),
                    "shield_reason": info.get("shield_reason", ""),
                    "eps": info.get("eps", np.nan),
                    "inflate": info.get("inflate", np.nan),
                    "ctx_id": info.get("ctx_id", -1),
                },
            )
        return True


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--preset", default="", help="e.g. merge_discrete_default")

    # IMPORTANT: default None so preset can fill, CLI can override
    ap.add_argument("--env", default=None, help="merge-v0 or highway-v0")
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--p_stay", type=float, default=None)

    ap.add_argument("--no_tier2", action="store_true")
    ap.add_argument("--no_conformal", action="store_true")
    ap.add_argument("--no_mpc", action="store_true")

    ap.add_argument("--run_dir", default="")
    return ap


def _load_preset(name: str) -> Dict[str, Any]:
    if not name:
        return {}
    if get_preset is None:
        raise RuntimeError(
            "Preset requested but scripts/presets.py:get_preset not found. "
            "Either add get_preset or remove --preset."
        )
    cfg = get_preset(name)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Preset {name} did not return a dict.")
    return cfg


def main() -> None:
    ap = _parser()
    args = ap.parse_args()

    preset_cfg: Dict[str, Any] = {}
    if args.preset:
        preset_cfg = _load_preset(args.preset)

    # --- Apply preset values ONLY where CLI did not provide a value ---
    # Support both 'env' and 'env_id' in preset
    preset_env = preset_cfg.get("env_id", preset_cfg.get("env", None))
    if args.env is None and preset_env is not None:
        args.env = preset_env

    if args.total_steps is None:
        args.total_steps = int(preset_cfg.get("total_steps", 200_000))
    if args.seed is None:
        args.seed = int(preset_cfg.get("seed", 0))

    ns = preset_cfg.get("nonstationarity", {}) or {}
    if args.p_stay is None:
        args.p_stay = float(ns.get("p_stay", 0.8))

    # Normalize env id (this is the key fix for your error)
    args.env = _normalize_env_id(args.env)

    # Diagnostics so we can SEE what's being used
    print("\n=== TRAIN DISCRETE DIAGNOSTICS ===")
    print("algo:", args.algo)
    print("preset:", args.preset)
    print("env:", args.env)
    print("seed:", args.seed)
    print("total_steps:", args.total_steps)
    print("p_stay:", args.p_stay)
    print("no_mpc:", args.no_mpc, "no_conformal:", args.no_conformal)
    print("=================================\n")

    # Safety params
    safety_kwargs = (preset_cfg.get("safety", {}) if preset_cfg else {}) or {}
    if not safety_kwargs:
        safety_kwargs = {"horizon_n": 10, "epsilon": 0.5}
    safety_params = SafetyParams(**safety_kwargs)

    # Run directory
    run_name = args.run_dir or f"{args.env}_discrete_{args.algo}_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))

    # Make env (now guaranteed merge-v0/highway-v0)
    env, _, _ = make_env(
        args.env,
        args.seed,
        "discrete",
        args.p_stay,
        args.no_mpc,
        args.no_conformal,
        safety_params,
    )

    save_json(
        os.path.join(run_dir, "config.json"),
        {**vars(args), "preset_cfg": preset_cfg, "safety_params": safety_params.__dict__},
    )

    sb3_logger = sb3_configure(run_dir, ["stdout", "csv", "tensorboard"])

    hp = (preset_cfg.get("sb3", {}) if preset_cfg else {}) or {}

    if args.algo == "dqn":
        dqn_hp = hp.get("dqn", {}) if isinstance(hp, dict) else {}
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=dqn_hp.get("learning_rate", 3e-4),
            buffer_size=dqn_hp.get("buffer_size", 100_000),
            learning_starts=dqn_hp.get("learning_starts", 10_000),
            batch_size=dqn_hp.get("batch_size", 64),
            gamma=dqn_hp.get("gamma", 0.99),
            train_freq=dqn_hp.get("train_freq", 4),
            target_update_interval=dqn_hp.get("target_update_interval", 1_000),
            verbose=1,
            seed=args.seed,
        )
    else:
        ppo_hp = hp.get("ppo", {}) if isinstance(hp, dict) else {}
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_hp.get("learning_rate", 3e-4),
            n_steps=ppo_hp.get("n_steps", 2048),
            batch_size=ppo_hp.get("batch_size", 64),
            gamma=ppo_hp.get("gamma", 0.99),
            gae_lambda=ppo_hp.get("gae_lambda", 0.95),
            ent_coef=ppo_hp.get("ent_coef", 0.0),
            verbose=1,
            seed=args.seed,
        )

    model.set_logger(sb3_logger)

    model.learn(
        total_timesteps=int(args.total_steps),
        callback=TrainLoggerCallback(run_dir),
        log_interval=1,
        progress_bar=True,
    )

    model.save(os.path.join(run_dir, "models", "final_model"))
    env.close()
    print(f"Saved model to {os.path.join(run_dir, 'models', 'final_model.zip')}")


if __name__ == "__main__":
    main()