from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.objective import (
    load_objective_data,
    make_schedule_dataframe,
)
from src.baselines import (
    random_baseline,
    greedy_appeal_baseline,
)
from src.optimizers.simulated_annealing import simulated_annealing


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def flatten_result(result: Dict[str, Any], seed: int | None = None) -> Dict[str, Any]:
    row = {
        "method": result["method"],
        "seed": seed,
        "best_score": result["best_score"],
        "runtime": result["runtime"],
    }

    if "iterations" in result:
        row["iterations"] = result["iterations"]
        row["objective_evaluations"] = result["iterations"] + 1
    else:
        row["iterations"] = 0
        row["objective_evaluations"] = 1

    row.update(result["components"])
    return row


def summarize_results(runs_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs_df
        .groupby("method")
        .agg(
            n_runs=("best_score", "count"),
            mean_score=("best_score", "mean"),
            std_score=("best_score", "std"),
            min_score=("best_score", "min"),
            max_score=("best_score", "max"),
            median_score=("best_score", "median"),
            mean_runtime=("runtime", "mean"),
            mean_budget_usage=("budget_usage", "mean"),
            mean_appeal=("appeal", "mean"),
            mean_diversity=("diversity", "mean"),
            mean_flow=("flow", "mean"),
            mean_conflict=("conflict", "mean"),
            mean_budget_penalty=("budget_penalty", "mean"),
            mean_duplicate_penalty=("duplicate_penalty", "mean"),
            mean_headliner_penalty=("headliner_penalty", "mean"),
        )
        .reset_index()
    )

    intervals = (
        runs_df
        .groupby("method")["best_score"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "score_q025", 0.975: "score_q975"})
    )

    summary = summary.merge(intervals, on="method", how="left")
    return summary


def plot_score_distribution(runs_df: pd.DataFrame) -> None:
    methods = runs_df["method"].unique().tolist()

    data = [
        runs_df.loc[runs_df["method"] == method, "best_score"].values
        for method in methods
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=methods)
    plt.title("Score distribution over multiple runs")
    plt.ylabel("Best score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "alena_score_distribution.png", dpi=200)
    plt.close()


def plot_sa_convergence(sa_histories: List[Dict[str, Any]]) -> None:
    if not sa_histories:
        return

    min_len = min(len(item["history"]) for item in sa_histories)

    histories = np.array([
        item["history"][:min_len]
        for item in sa_histories
    ])

    mean_history = histories.mean(axis=0)

    plt.figure(figsize=(9, 5))

    for item in sa_histories:
        plt.plot(
            item["history"][:min_len],
            alpha=0.25,
            linewidth=1,
        )

    plt.plot(
        mean_history,
        linewidth=2.5,
        label="Mean SA best score",
    )

    plt.title("Simulated Annealing convergence over multiple seeds")
    plt.xlabel("Iteration")
    plt.ylabel("Best score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sa_convergence_multiseed.png", dpi=200)
    plt.close()


def run_alena_experiments() -> None:
    artists_df, similarity_df, config = load_objective_data()

    n_runs = int(config.get("experiments", {}).get("n_runs", 10))
    random_seed = int(config.get("experiments", {}).get("random_seed", 42))

    seeds = list(range(random_seed, random_seed + n_runs))

    rows = []
    sa_histories = []

    print("Running random baseline over seeds...")

    for seed in seeds:
        result = random_baseline(
            artists_df=artists_df,
            similarity_df=similarity_df,
            config=config,
            random_seed=seed,
        )
        rows.append(flatten_result(result, seed=seed))

    print("Running greedy baseline once...")

    greedy_result = greedy_appeal_baseline(
        artists_df=artists_df,
        similarity_df=similarity_df,
        config=config,
    )
    rows.append(flatten_result(greedy_result, seed=None))

    print("Running Simulated Annealing from random initialization over seeds...")

    best_sa_result = None

    for seed in seeds:
        print(f"SA seed = {seed}")

        result = simulated_annealing(
            artists_df=artists_df,
            similarity_df=similarity_df,
            config=config,
            initial_solution=None,
            random_seed=seed,
        )

        # На всякий случай фиксируем имя метода.
        result["method"] = "simulated_annealing"

        rows.append(flatten_result(result, seed=seed))

        sa_histories.append(
            {
                "seed": seed,
                "history": result["history"],
                "current_history": result["current_history"],
            }
        )

        if best_sa_result is None or result["best_score"] > best_sa_result["best_score"]:
            best_sa_result = result

    runs_df = pd.DataFrame(rows)
    summary_df = summarize_results(runs_df)

    runs_path = RESULTS_DIR / "alena_experiment_runs.csv"
    summary_path = RESULTS_DIR / "alena_experiment_summary.csv"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if best_sa_result is not None:
        best_schedule_df = make_schedule_dataframe(
            best_sa_result["best_solution"],
            artists_df,
            config,
        )

        best_schedule_df.to_csv(
            RESULTS_DIR / "best_schedule_sa_multiseed.csv",
            index=False,
        )

        best_history_df = pd.DataFrame(
            {
                "iteration": np.arange(len(best_sa_result["history"])),
                "best_score": best_sa_result["history"],
                "current_score": best_sa_result["current_history"],
            }
        )

        best_history_df.to_csv(
            RESULTS_DIR / "best_sa_history_multiseed.csv",
            index=False,
        )

    plot_score_distribution(runs_df)
    plot_sa_convergence(sa_histories)

    print("\nExperiment runs saved to:")
    print(runs_path)

    print("\nSummary saved to:")
    print(summary_path)

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    run_alena_experiments()