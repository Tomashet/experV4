# scripts/make_camera_ready_figures.py
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@dataclass(frozen=True)
class RunMeta:
    path: str
    env: str
    space: str          # discrete/continuous
    algo: str
    seed: int
    p_stay: float
    condition: str      # stationary/nonstationary
    safe: str           # safeon/safeoff (derived)
    name: str


def parse_run_dir(run_dir: str) -> Optional[RunMeta]:
    # Expected:
    # runs/{env}_{space}_{algo}_seed{seed}_pst{p_stay:.2f}_{condition}_safe{on/off}
    base = os.path.basename(run_dir)
    parts = base.split("_")
    if len(parts) < 7:
        return None
    try:
        env = parts[0]
        space = parts[1]  # discrete/continuous
        algo = parts[2]
        seed = int(parts[3].replace("seed", ""))
        p_stay = float(parts[4].replace("pst", ""))
        condition = parts[5]
        safe_tag = parts[6]  # e.g., safeon
        safe = "on" if "safeon" in safe_tag else ("off" if "safeoff" in safe_tag else safe_tag.replace("safe",""))
    except Exception:
        return None
    return RunMeta(path=run_dir, env=env, space=space, algo=algo, seed=seed, p_stay=p_stay,
                   condition=condition, safe=safe, name=base)


def collect_runs(runs_dir: str) -> List[RunMeta]:
    run_dirs = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    metas = []
    for r in run_dirs:
        m = parse_run_dir(r)
        if m is not None:
            metas.append(m)
    return metas


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window, min_periods=1).mean().to_numpy()


def plot_group_curves(
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    max_points: int = 2000,
) -> None:
    plt.figure(figsize=(7.2, 4.8))

    for label, curves in groups.items():
        # interpolate to common grid over overlap
        mins = [c[0].min() for c in curves]
        maxs = [c[0].max() for c in curves]
        tmin, tmax = float(max(mins)), float(min(maxs))
        if tmax <= tmin:
            # fallback: plot individual lines
            for x, y in curves:
                plt.plot(x, y, alpha=0.25)
            continue
        grid = np.linspace(tmin, tmax, max_points)
        ys = [np.interp(grid, x, y) for x, y in curves]
        Y = np.stack(ys, axis=0)
        mean = Y.mean(axis=0)
        se = Y.std(axis=0, ddof=1) / np.sqrt(max(Y.shape[0], 1))
        plt.plot(grid, mean, linewidth=2.5, label=label)
        plt.fill_between(grid, mean - 1.96 * se, mean + 1.96 * se, alpha=0.15)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(groups) > 1:
        plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="paper_figures")
    ap.add_argument("--window", type=int, default=2000, help="rolling window for violation rate (timesteps)")
    ap.add_argument("--max_steps", type=int, default=200000, help="max timesteps for plots (clip)")
    args = ap.parse_args()

    runs = collect_runs(args.runs_dir)
    if len(runs) == 0:
        raise SystemExit(f"No parsable run directories found under: {args.runs_dir}")

    # 1) Reward curves: stationary vs nonstationary (safe=off by default)
    reward_groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for m in runs:
        prog = safe_read_csv(os.path.join(m.path, "progress.csv"))
        if prog is None:
            continue
        if "time/total_timesteps" not in prog.columns or "rollout/ep_rew_mean" not in prog.columns:
            continue
        x = prog["time/total_timesteps"].to_numpy(dtype=float)
        y = prog["rollout/ep_rew_mean"].to_numpy(dtype=float)
        mask = x <= args.max_steps
        x, y = x[mask], y[mask]
        label = f"{m.condition} | {m.algo.upper()} | safe={m.safe}"
        reward_groups.setdefault(label, []).append((x, y))

    plot_group_curves(
        reward_groups,
        xlabel="Timesteps",
        ylabel="Episode reward (mean)",
        title="Learning curves across conditions (mean ± 95% CI across seeds)",
        out_path=os.path.join(args.out_dir, "fig_reward_curves.png"),
    )

    # 2) Violation curves from train_monitor.csv (0/1 per step)
    vio_groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for m in runs:
        mon = safe_read_csv(os.path.join(m.path, "train_monitor.csv"))
        if mon is None:
            continue
        if "timestep" not in mon.columns or "violation" not in mon.columns:
            continue
        x = mon["timestep"].to_numpy(dtype=float)
        v = mon["violation"].to_numpy(dtype=float)
        mask = x <= args.max_steps
        x, v = x[mask], v[mask]
        y = rolling_mean(v, args.window)
        label = f"{m.condition} | {m.algo.upper()} | safe={m.safe}"
        vio_groups.setdefault(label, []).append((x, y))

    plot_group_curves(
        vio_groups,
        xlabel="Timesteps",
        ylabel=f"Violation rate (rolling mean, window={args.window})",
        title="Safety violations during training (mean ± 95% CI across seeds)",
        out_path=os.path.join(args.out_dir, "fig_violation_curves.png"),
    )

    # 3) Context switching timeline for any nonstationary run available
    example = next((m for m in runs if m.condition == "nonstationary"), None)
    if example is not None:
        mon = safe_read_csv(os.path.join(example.path, "train_monitor.csv"))
        if mon is not None and "timestep" in mon.columns and "ctx_id" in mon.columns:
            df = mon[["timestep", "ctx_id"]].dropna()
            df = df[df["timestep"] <= min(args.max_steps, df["timestep"].max())]
            plt.figure(figsize=(9.0, 2.8))
            plt.plot(df["timestep"], df["ctx_id"], drawstyle="steps-post")
            plt.xlabel("Timesteps")
            plt.ylabel("Context ID")
            plt.title(f"Markov context switching (example: {example.name})")
            plt.tight_layout()
            os.makedirs(args.out_dir, exist_ok=True)
            plt.savefig(os.path.join(args.out_dir, "fig_context_timeline.png"), dpi=250)
            plt.close()

    print(f"Wrote figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
