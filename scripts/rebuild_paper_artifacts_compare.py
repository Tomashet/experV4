from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read CSV: {path} ({exc})")
        return None


def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = len(vals)
    if n <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(n))


def mean_tail(series: pd.Series, tail_frac: float = 0.10) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan")
    k = max(1, int(math.ceil(len(vals) * tail_frac)))
    return float(vals.iloc[-k:].mean())


# ============================================================
# Run parsing
# ============================================================

@dataclass(frozen=True)
class RunMeta:
    run_dir: str
    run_name: str
    env: str
    space: str
    algo: str
    seed: int
    p_stay: float
    regime: str
    safe: str


RUN_RE = re.compile(
    r"^(?P<env>[^_]+)_(?P<space>[^_]+)_(?P<algo>[^_]+)_seed(?P<seed>\d+)_pst(?P<pstay>[0-9.]+)_(?P<regime>stationary|nonstationary)_(?P<safe>safeon|safeoff)$",
    re.IGNORECASE,
)


def parse_run_dir(run_dir: str) -> Optional[RunMeta]:
    run_name = os.path.basename(run_dir)
    match = RUN_RE.match(run_name)
    if match is None:
        return None

    gd = match.groupdict()

    return RunMeta(
        run_dir=run_dir,
        run_name=run_name,
        env=gd["env"],
        space=gd["space"],
        algo=gd["algo"].lower(),
        seed=int(gd["seed"]),
        p_stay=float(gd["pstay"]),
        regime=gd["regime"].lower(),
        safe="on" if gd["safe"].lower() == "safeon" else "off",
    )


def collect_runs(runs_dir: str) -> List[RunMeta]:
    run_dirs = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    metas: List[RunMeta] = []

    for rd in run_dirs:
        meta = parse_run_dir(rd)
        if meta is not None:
            metas.append(meta)
        else:
            print(f"[INFO] Skipping unrecognized folder name: {os.path.basename(rd)}")

    metas.sort(key=lambda m: (m.algo, m.p_stay, m.regime, m.safe, m.seed, m.run_name))
    return metas


# ============================================================
# Column inference
# ============================================================

def infer_reward_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "rollout/ep_rew_mean",
        "eval/mean_reward",
        "episode_reward",
        "reward",
        "ep_rew_mean",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_timestep_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "time/total_timesteps",
        "total_timesteps",
        "timestep",
        "timesteps",
        "steps",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_violation_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "violation",
        "violations",
        "safety_violation",
        "constraint_violation",
        "budget_violation",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# Metric extraction
# ============================================================

def final_reward_mean(progress: Optional[pd.DataFrame], tail_frac: float = 0.10) -> float:
    if progress is None:
        return float("nan")

    reward_col = infer_reward_column(progress)
    if reward_col is None:
        return float("nan")

    return mean_tail(progress[reward_col], tail_frac=tail_frac)


def episode_violation_rate(train_monitor: Optional[pd.DataFrame]) -> float:
    if train_monitor is None:
        return float("nan")

    violation_col = infer_violation_column(train_monitor)
    if violation_col is None:
        return float("nan")

    v = pd.to_numeric(train_monitor[violation_col], errors="coerce")

    if "episode" in train_monitor.columns:
        tmp = train_monitor.copy()
        tmp["_viol"] = v
        out = tmp.groupby("episode")["_viol"].max().mean()
        return float(out)

    return float(v.mean())


def load_reward_curve(progress: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if progress is None:
        return None

    t_col = infer_timestep_column(progress)
    r_col = infer_reward_column(progress)

    if t_col is None or r_col is None:
        return None

    out = progress[[t_col, r_col]].copy()
    out.columns = ["timesteps", "reward"]
    out["timesteps"] = pd.to_numeric(out["timesteps"], errors="coerce")
    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
    out = out.dropna().sort_values("timesteps")

    if out.empty:
        return None

    return out


# ============================================================
# Aggregation
# ============================================================

def build_all_run_metrics(runs: List[RunMeta]) -> pd.DataFrame:
    rows: List[Dict] = []

    for meta in runs:
        progress_path = os.path.join(meta.run_dir, "progress.csv")
        train_monitor_path = os.path.join(meta.run_dir, "train_monitor.csv")

        progress = safe_read_csv(progress_path)
        train_monitor = safe_read_csv(train_monitor_path)

        row = {
            "run_name": meta.run_name,
            "env": meta.env,
            "space": meta.space,
            "algo": meta.algo,
            "seed": meta.seed,
            "p_stay": meta.p_stay,
            "regime": meta.regime,
            "safe": meta.safe,
            "final_reward_mean": final_reward_mean(progress),
            "violation_rate": episode_violation_rate(train_monitor),
            "has_progress": progress is not None,
            "has_train_monitor": train_monitor is not None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    if all_runs_df.empty:
        return all_runs_df.copy()

    summary = (
        all_runs_df.groupby(["algo", "p_stay", "regime", "safe"], as_index=False)
        .agg(
            final_reward_mean_mean=("final_reward_mean", "mean"),
            final_reward_mean_se=("final_reward_mean", sem),
            violation_rate_mean=("violation_rate", "mean"),
            violation_rate_se=("violation_rate", sem),
            n=("seed", "nunique"),
        )
        .sort_values(["algo", "p_stay", "regime", "safe"])
        .reset_index(drop=True)
    )
    return summary


def filter_complete_comparison_groups(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only groups that have both safe=off and safe=on.
    This excludes incomplete groups such as SAC safe-off-only runs.
    """
    if summary_df.empty:
        return summary_df.copy()

    keep_chunks: List[pd.DataFrame] = []

    for _, g in summary_df.groupby(["algo", "p_stay", "regime"], as_index=False):
        safe_modes = set(g["safe"].astype(str).tolist())
        if {"off", "on"}.issubset(safe_modes):
            keep_chunks.append(g)

    if not keep_chunks:
        return summary_df.iloc[0:0].copy()

    return (
        pd.concat(keep_chunks, axis=0)
        .sort_values(["algo", "p_stay", "regime", "safe"])
        .reset_index(drop=True)
    )


# ============================================================
# LaTeX table
# ============================================================

def summary_to_latex(summary_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Setting & Safety & Violation rate $\downarrow$ & Return $\uparrow$ \\")
    lines.append(r"\midrule")

    for _, r in summary_df.iterrows():
        algo = str(r["algo"]).upper()
        setting = f"{r['regime']} ($p_{{\\mathrm{{stay}}}}={r['p_stay']:.2f}$)"
        safety = str(r["safe"])

        vio = "--"
        if pd.notna(r["violation_rate_mean"]):
            vio = f"{100.0 * r['violation_rate_mean']:.2f}\\%"

        ret = "--"
        if pd.notna(r["final_reward_mean_mean"]):
            ret = f"{r['final_reward_mean_mean']:.3f}"

        lines.append(f"{algo} & {setting} & {safety} & {vio} & {ret} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Training-time comparison across algorithms, dynamics regimes, and safety settings. "
        r"Violation rate is computed from train\_monitor logs and reported as a percentage. "
        r"Return is the mean of the final 10\% of reward checkpoints, averaged over seeds.}"
    )
    lines.append(r"\label{tab:main_results_compare}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# Plot labeling
# ============================================================

def label_for_row(row: pd.Series) -> str:
    """
    Output examples:
        DQN nonstat
        DQN stat
        PPO nonstat
        PPO stat
    """
    algo = str(row["algo"]).strip().upper()
    regime = str(row["regime"]).strip().lower()
    regime_label = "nonstat" if regime == "nonstationary" else "stat"
    return f"{algo} {regime_label}"


def build_plot_labels(summary_df: pd.DataFrame) -> List[str]:
    """
    Stable ordering for comparison figures.
    Only DQN/PPO are included in the paper comparison plots.
    """
    labels: List[str] = []
    order = [
        ("dqn", "nonstationary"),
        ("dqn", "stationary"),
        ("ppo", "nonstationary"),
        ("ppo", "stationary"),
    ]

    for algo, regime in order:
        sub = summary_df[
            (summary_df["algo"] == algo) &
            (summary_df["regime"] == regime)
        ]
        if not sub.empty:
            regime_label = "nonstat" if regime == "nonstationary" else "stat"
            labels.append(f"{algo.upper()} {regime_label}")

    return labels


# ============================================================
# Plotting
# ============================================================

def plot_bar_metric(
    summary_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    if summary_df.empty:
        print(f"[WARN] No data available for plot: {out_path}")
        return

    plot_df = summary_df.copy()
    plot_df["label"] = plot_df.apply(label_for_row, axis=1)

    labels = build_plot_labels(plot_df)
    x = np.arange(len(labels))
    width = 0.36

    safe_off: List[float] = []
    safe_on: List[float] = []

    for label in labels:
        sub = plot_df[plot_df["label"] == label]
        off_row = sub[sub["safe"] == "off"]
        on_row = sub[sub["safe"] == "on"]

        safe_off.append(float(off_row.iloc[0][metric_col]) if len(off_row) else np.nan)
        safe_on.append(float(on_row.iloc[0][metric_col]) if len(on_row) else np.nan)

    safe_off_plot = np.array(safe_off, dtype=float)
    safe_on_true = np.array(safe_on, dtype=float)

    ymax_candidates = np.concatenate([
        safe_off_plot[np.isfinite(safe_off_plot)],
        safe_on_true[np.isfinite(safe_on_true)],
    ])
    ymax = float(np.max(ymax_candidates)) if len(ymax_candidates) else 1.0
    label_offset = max(0.015, 0.025 * max(ymax, 0.1))

    plt.figure(figsize=(10.0, 5.0))
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Draw true bar heights only
    plt.bar(x - width / 2, safe_off_plot, width, label="safe off", edgecolor="black")
    plt.bar(x + width / 2, safe_on_true, width, label="safe on", edgecolor="black")

    # Label safe-off bars
    for xi, y in zip(x - width / 2, safe_off_plot):
        if np.isfinite(y):
            plt.text(xi, y + label_offset, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

    # Label safe-on bars; if zero, place label just above baseline
    for xi, y in zip(x + width / 2, safe_on_true):
        if np.isfinite(y):
            if y == 0:
                plt.text(xi, 0.02, "0.00", ha="center", va="bottom", fontsize=9)
            else:
                plt.text(xi, y + label_offset, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if "violation" in title.lower():
        plt.ylim(0, max(1.25, ymax + 0.12))
    else:
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def interpolate_mean_curve(curves: List[pd.DataFrame], n_grid: int = 200) -> Optional[pd.DataFrame]:
    if not curves:
        return None

    mins = [c["timesteps"].min() for c in curves]
    maxs = [c["timesteps"].max() for c in curves]
    tmin = max(mins)
    tmax = min(maxs)

    if tmax <= tmin:
        return None

    grid = np.linspace(tmin, tmax, n_grid)
    ys = []

    for c in curves:
        ys.append(np.interp(grid, c["timesteps"].to_numpy(), c["reward"].to_numpy()))

    mean_y = np.mean(np.stack(ys, axis=0), axis=0)
    return pd.DataFrame({"timesteps": grid, "reward": mean_y})


def plot_mean_reward_curves(runs: List[RunMeta], algo: str, out_path: str) -> None:
    groups: Dict[str, List[pd.DataFrame]] = {
        "nonstationary_off": [],
        "nonstationary_on": [],
        "stationary_off": [],
        "stationary_on": [],
    }

    for meta in runs:
        if meta.algo != algo:
            continue

        progress = safe_read_csv(os.path.join(meta.run_dir, "progress.csv"))
        curve = load_reward_curve(progress)
        if curve is None:
            continue

        key = f"{meta.regime}_{meta.safe}"
        if key in groups:
            groups[key].append(curve)

    plt.figure(figsize=(9.2, 5.4))
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    order = [
        ("nonstationary_off", "nonstat safe off"),
        ("nonstationary_on", "nonstat safe on"),
        ("stationary_off", "stat safe off"),
        ("stationary_on", "stat safe on"),
    ]

    for key, label in order:
        mean_curve = interpolate_mean_curve(groups[key])
        if mean_curve is not None:
            plt.plot(mean_curve["timesteps"], mean_curve["reward"], label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean episode reward")
    plt.title(f"{algo.upper()} reward curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild paper comparison artifacts from run folders.")
    parser.add_argument("--runs_dir", default="runs", help="Directory containing run subfolders.")
    parser.add_argument("--graphs_dir", default="Graphs", help="Output directory for generated figures.")
    parser.add_argument("--all_runs_csv", default="all_run_metrics.csv", help="Output CSV with per-run metrics.")
    parser.add_argument("--summary_csv", default="main_table_p05_summary.csv", help="Output CSV with grouped summary.")
    parser.add_argument("--summary_tex", default="main_table_p05.tex", help="Output LaTeX table.")
    args = parser.parse_args()

    ensure_dir(args.graphs_dir)

    runs = collect_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No parsable run directories found under {args.runs_dir}")

    print(f"Scanning {len(runs)} run folders under {args.runs_dir} ...")

    all_runs_df = build_all_run_metrics(runs)
    if all_runs_df.empty:
        raise SystemExit("No usable run data found.")

    all_runs_df.to_csv(args.all_runs_csv, index=False)
    print(f"Saved {args.all_runs_csv}")

    summary_df = build_summary(all_runs_df)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"Saved {args.summary_csv}")

    comparison_df = filter_complete_comparison_groups(summary_df)

    with open(args.summary_tex, "w", encoding="utf-8") as f:
        f.write(summary_to_latex(comparison_df))
    print(f"Saved {args.summary_tex}")

    plot_df = comparison_df.copy()
    plot_df["violation_rate_percent"] = 100.0 * plot_df["violation_rate_mean"]

    plot_bar_metric(
        summary_df=plot_df,
        metric_col="violation_rate_percent",
        ylabel="Violation rate (%)",
        title="Logged training-time violation rate",
        out_path=os.path.join(args.graphs_dir, "fig_violation_bars.png"),
    )
    print(f"Saved {os.path.join(args.graphs_dir, 'fig_violation_bars.png')}")

    plot_bar_metric(
        summary_df=plot_df,
        metric_col="final_reward_mean_mean",
        ylabel="Final reward mean",
        title="Final training reward",
        out_path=os.path.join(args.graphs_dir, "fig_return_bars.png"),
    )
    print(f"Saved {os.path.join(args.graphs_dir, 'fig_return_bars.png')}")

    plot_mean_reward_curves(
        runs=runs,
        algo="dqn",
        out_path=os.path.join(args.graphs_dir, "fig_dqn_reward_curves.png"),
    )
    print(f"Saved {os.path.join(args.graphs_dir, 'fig_dqn_reward_curves.png')}")

    plot_mean_reward_curves(
        runs=runs,
        algo="ppo",
        out_path=os.path.join(args.graphs_dir, "fig_ppo_reward_curves.png"),
    )
    print(f"Saved {os.path.join(args.graphs_dir, 'fig_ppo_reward_curves.png')}")


if __name__ == "__main__":
    main()