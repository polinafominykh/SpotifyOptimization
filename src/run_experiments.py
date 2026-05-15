import copy
import time
from pathlib import Path

import pandas as pd

from objective import (
    load_objective_data,
    make_schedule_dataframe,
)

from optimizers.genetic_algorithm import run_genetic_algorithm
from optimizers.particle_swarm import run_particle_swarm


def make_config_for_run(config, run_id):
    """
    Для каждого запуска меняем random_seed,
    чтобы стохастические методы давали разные результаты.
    """
    config_run = copy.deepcopy(config)

    base_seed = config["experiments"].get("random_seed", 42)
    config_run["experiments"]["random_seed"] = base_seed + run_id

    return config_run


def extract_result_row(method_name, run_id, optimizer_result, runtime):
    """
    Делает одну строку для итоговой таблицы.
    """
    components = optimizer_result["best_components"]

    return {
        "method": method_name,
        "run_id": run_id,
        "best_score": optimizer_result["best_score"],
        "appeal": components.get("appeal"),
        "diversity": components.get("diversity"),
        "flow": components.get("flow"),
        "conflict": components.get("conflict"),
        "budget_penalty": components.get("budget_penalty"),
        "duplicate_penalty": components.get("duplicate_penalty"),
        "headliner_penalty": components.get("headliner_penalty"),
        "invalid_artist_penalty": components.get("invalid_artist_penalty"),
        "total_cost": components.get("total_cost"),
        "budget": components.get("budget"),
        "budget_usage": components.get("budget_usage"),
        "runtime_seconds": runtime,
    }


def save_best_schedule(method_name, best_solution, artists_df, config, results_dir):
    """
    Сохраняет лучшее расписание метода.
    """
    schedule_df = make_schedule_dataframe(
        best_solution,
        artists_df,
        config,
    )

    output_path = results_dir / f"best_schedule_{method_name.lower()}.csv"
    schedule_df.to_csv(output_path, index=False)

    return output_path


def main():
    artists_df, similarity_df, config = load_objective_data()

    root_dir = Path(__file__).resolve().parents[1]
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_runs = config["experiments"].get("n_runs", 3)

    methods = {
        "GA": run_genetic_algorithm,
        "PSO": run_particle_swarm,
    }

    all_results = []
    all_histories = []
    best_by_method = {}

    for run_id in range(n_runs):
        print("=" * 70)
        print(f"Run {run_id + 1}/{n_runs}")
        print("=" * 70)

        config_run = make_config_for_run(config, run_id)

        for method_name, method_function in methods.items():
            print(f"\nRunning {method_name}...")

            start_time = time.time()

            optimizer_result = method_function(
                artists_df,
                similarity_df,
                config_run,
            )

            runtime = time.time() - start_time

            print(f"{method_name} best score: {optimizer_result['best_score']:.4f}")
            print(f"{method_name} runtime: {runtime:.2f} sec")

            row = extract_result_row(
                method_name,
                run_id,
                optimizer_result,
                runtime,
            )

            all_results.append(row)

            history = optimizer_result["history"].copy()
            history["method"] = method_name
            history["run_id"] = run_id
            all_histories.append(history)

            if method_name not in best_by_method:
                best_by_method[method_name] = optimizer_result
            else:
                if optimizer_result["best_score"] > best_by_method[method_name]["best_score"]:
                    best_by_method[method_name] = optimizer_result

    results_df = pd.DataFrame(all_results)

    results_path = results_dir / "experiment_results_ga_pso.csv"
    results_df.to_csv(results_path, index=False)

    histories_df = pd.concat(all_histories, ignore_index=True)

    histories_path = results_dir / "convergence_history_ga_pso.csv"
    histories_df.to_csv(histories_path, index=False)

    for method_name, method_result in best_by_method.items():
        output_path = save_best_schedule(
            method_name,
            method_result["best_solution"],
            artists_df,
            config,
            results_dir,
        )

        print(f"Best schedule for {method_name} saved to: {output_path}")

    summary_df = (
        results_df
        .groupby("method")
        .agg(
            best_score_max=("best_score", "max"),
            best_score_mean=("best_score", "mean"),
            best_score_std=("best_score", "std"),
            runtime_mean=("runtime_seconds", "mean"),
            budget_usage_mean=("budget_usage", "mean"),
        )
        .reset_index()
        .sort_values(by="best_score_max", ascending=False)
    )

    summary_path = results_dir / "experiment_summary_ga_pso.csv"
    summary_df.to_csv(summary_path, index=False)

    best_method_name = max(
        best_by_method,
        key=lambda name: best_by_method[name]["best_score"],
    )

    final_schedule_df = make_schedule_dataframe(
        best_by_method[best_method_name]["best_solution"],
        artists_df,
        config,
    )

    final_schedule_path = results_dir / "best_schedule_final_ga_pso.csv"
    final_schedule_df.to_csv(final_schedule_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENTS FINISHED")
    print("=" * 70)

    print("\nSummary:")
    print(summary_df)

    print("\nBest method among GA and PSO:")
    print(best_method_name)

    print("\nSaved files:")
    print(results_path)
    print(histories_path)
    print(summary_path)
    print(final_schedule_path)


if __name__ == "__main__":
    main()