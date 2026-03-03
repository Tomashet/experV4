# scripts/summarize_results.py
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
    space: str
    algo: str
    seed: int
    p_stay: float
    condition: str
    safe: str
    name: str


def parse_run_dir(run_dir: str) -> Optional[RunMeta]:
    base = os.path.basename(run_dir)
    parts = base.split("_")
    if len(parts) < 7:
        return None
    try:
        env = parts[0]
        space = parts[1]
        algo = parts[2]
        seed = int(parts[3].replace("seed", ""))
        p_stay = float(parts[4].replace("pst", ""))
        condition = parts[5]
        safe_tag = parts[6]
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


def episode_violation_rate(train_monitor: pd.DataFrame) -> float:
    # If monitor has 'episode' and per-step 'violation', mark an episode violated if any step violated.
    if "episode" in train_monitor.columns and "violation" in train_monitor.columns:
        ep = train_monitor.groupby("episode")["violation"].max()
        return float(ep.mean())
    # fallback: per-step violation mean
    if "violation" in train_monitor.columns:
        return float(train_monitor["violation"].astype(float).mean())
    return float("nan")


def final_reward_mean(progress: pd.DataFrame, tail_frac: float = 0.1) -> float:
    if "rollout/ep_rew_mean" not in progress.columns:
        return float("nan")
    y = progress["rollout/ep_rew_mean"].astype(float)
    n = len(y)
    k = max(int(np.ceil(n * tail_frac)), 1)
    return float(y.iloc[-k:].mean())


def latex_table(df: pd.DataFrame) -> str:
    # expects columns: condition, algo, safe, reward_mean, reward_se, vio_mean, vio_se
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lllccl}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Algo & Safe & Return $\uparrow$ & Violation $\downarrow$ \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        ret = f"{r['reward_mean']:.1f} $\pm$ {r['reward_se']:.1f}" if np.isfinite(r['reward_mean']) else "--"
        vio = f"{100*r['vio_mean']:.2f}\% $\pm$ {100*r['vio_se']:.2f}\%" if np.isfinite(r['vio_mean']) else "--"
        lines.append(f"{r['condition']} & {r['algo'].upper()} & {r['safe']} & {ret} & {vio} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Final performance (mean $\pm$ s.e. over seeds). Return is the average of the last 10\% of training checkpoints; Violation is the episode violation rate (or per-step rate if episode IDs are unavailable).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_csv", default="paper_figures/results_summary.csv")
    ap.add_argument("--out_latex", default="paper_figures/results_table.tex")
    args = ap.parse_args()

    runs = collect_runs(args.runs_dir)
    if len(runs) == 0:
        raise SystemExit(f"No parsable run directories found under: {args.runs_dir}")

    rows = []
    for m in runs:
        prog = safe_read_csv(os.path.join(m.path, "progress.csv"))
        mon = safe_read_csv(os.path.join(m.path, "train_monitor.csv"))
        if prog is None and mon is None:
            continue
        reward = final_reward_mean(prog) if prog is not None else float("nan")
        vio = episode_violation_rate(mon) if mon is not None else float("nan")
        rows.append(dict(condition=m.condition, algo=m.algo, safe=m.safe, seed=m.seed,
                         reward=reward, violation=vio))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable progress.csv / train_monitor.csv found in runs.")

    # aggregate across seeds
    agg = df.groupby(["condition","algo","safe"], as_index=False).agg(
        reward_mean=("reward","mean"),
        reward_se=("reward", lambda x: x.std(ddof=1)/np.sqrt(max(len(x),1))),
        vio_mean=("violation","mean"),
        vio_se=("violation", lambda x: x.std(ddof=1)/np.sqrt(max(len(x),1))),
        n=("seed","nunique"),
    ).sort_values(["condition","algo","safe"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    agg.to_csv(args.out_csv, index=False)

    tex = latex_table(agg)
    with open(args.out_latex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_latex}")


if __name__ == "__main__":
    main()
