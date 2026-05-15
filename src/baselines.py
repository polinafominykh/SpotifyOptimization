import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd

try:
    from src.objective import (
        load_objective_data,
        evaluate_solution,
        get_solution_length,
        make_schedule_dataframe,
    )
except ModuleNotFoundError:
    from objective import (
        load_objective_data,
        evaluate_solution,
        get_solution_length,
        make_schedule_dataframe,
    )


def get_budget(config: dict) -> int:
    return int(config["festival"]["budget"])


def get_all_artist_ids(artists_df: pd.DataFrame) -> List[int]:
    return artists_df["artist_id"].astype(int).tolist()


def random_baseline(
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    config: dict,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Random baseline.

    Случайно выбирает нужное количество артистов без повторов.
    Этот baseline нужен как нижняя граница качества.
    """
    start_time = time.time()

    rng = np.random.default_rng(random_seed)

    solution_length = get_solution_length(config)
    artist_ids = get_all_artist_ids(artists_df)

    if len(artist_ids) < solution_length:
        raise ValueError(
            f"Not enough artists: {len(artist_ids)}. "
            f"Required: {solution_length}."
        )

    solution = rng.choice(
        artist_ids,
        size=solution_length,
        replace=False,
    ).astype(int).tolist()

    result = evaluate_solution(
        solution,
        artists_df,
        similarity_df,
        config,
    )

    runtime = time.time() - start_time

    return {
        "method": "random_baseline",
        "best_solution": solution,
        "best_score": result["score"],
        "history": [result["score"]],
        "components": result["components"],
        "runtime": runtime,
    }


def greedy_appeal_baseline(
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    config: dict,
) -> Dict[str, Any]:
    """
    Greedy baseline.

    Берет артистов с максимальным appeal_score.
    Если возможно, старается не выйти за бюджет.

    Это baseline показывает подход:
    "давайте просто возьмем самых привлекательных артистов".
    """
    start_time = time.time()

    solution_length = get_solution_length(config)
    budget = get_budget(config)

    candidates = artists_df.sort_values(
        by="appeal_score",
        ascending=False,
    ).copy()

    selected_artists = []
    current_cost = 0

    for _, row in candidates.iterrows():
        artist_id = int(row["artist_id"])
        artist_cost = float(row["cost"])

        if len(selected_artists) >= solution_length:
            break

        if current_cost + artist_cost <= budget:
            selected_artists.append(artist_id)
            current_cost += artist_cost

    # Если бюджет слишком жесткий и мы не набрали нужное число артистов,
    # добираем самых дешевых из оставшихся. Штраф за бюджет обработает objective.
    if len(selected_artists) < solution_length:
        selected_set = set(selected_artists)

        remaining = artists_df[
            ~artists_df["artist_id"].isin(selected_set)
        ].sort_values(by="cost", ascending=True)

        for _, row in remaining.iterrows():
            if len(selected_artists) >= solution_length:
                break

            selected_artists.append(int(row["artist_id"]))

    solution = arrange_headliners_to_evening_slots(
        selected_artists,
        artists_df,
        config,
    )

    result = evaluate_solution(
        solution,
        artists_df,
        similarity_df,
        config,
    )

    runtime = time.time() - start_time

    return {
        "method": "greedy_appeal_baseline",
        "best_solution": solution,
        "best_score": result["score"],
        "history": [result["score"]],
        "components": result["components"],
        "runtime": runtime,
    }


def arrange_headliners_to_evening_slots(
    solution: List[int],
    artists_df: pd.DataFrame,
    config: dict,
) -> List[int]:
    """
    Небольшое улучшение для greedy baseline:
    пытаемся поставить хедлайнеров в вечерние слоты.

    Это не полноценная оптимизация, а простое правило.
    """
    n_scenes = int(config["festival"]["n_scenes"])
    n_slots = int(config["festival"]["n_slots"])

    solution = solution.copy()

    artists_indexed = artists_df.set_index("artist_id")

    evening_slots = list(range(max(0, n_slots - 2), n_slots))
    evening_positions = []

    for slot_id in evening_slots:
        for scene_id in range(n_scenes):
            position = slot_id * n_scenes + scene_id
            evening_positions.append(position)

    headliner_positions = []

    for position, artist_id in enumerate(solution):
        artist_id = int(artist_id)

        if artist_id not in artists_indexed.index:
            continue

        if int(artists_indexed.loc[artist_id, "is_headliner"]) == 1:
            headliner_positions.append(position)

    non_evening_headliner_positions = [
        pos for pos in headliner_positions
        if pos not in evening_positions
    ]

    free_evening_positions = [
        pos for pos in evening_positions
        if pos not in headliner_positions
    ]

    for old_pos, new_pos in zip(non_evening_headliner_positions, free_evening_positions):
        solution[old_pos], solution[new_pos] = solution[new_pos], solution[old_pos]

    return solution


def run_baselines(random_seed: int = 42) -> pd.DataFrame:
    """
    Запускает оба baseline и сохраняет их расписания в results/.
    """
    artists_df, similarity_df, config = load_objective_data()

    baseline_results = []

    random_result = random_baseline(
        artists_df,
        similarity_df,
        config,
        random_seed=random_seed,
    )

    greedy_result = greedy_appeal_baseline(
        artists_df,
        similarity_df,
        config,
    )

    baseline_results.append(random_result)
    baseline_results.append(greedy_result)

    rows = []

    for result in baseline_results:
        row = {
            "method": result["method"],
            "best_score": result["best_score"],
            "runtime": result["runtime"],
        }

        row.update(result["components"])
        rows.append(row)

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        "results/baseline_results.csv",
        index=False,
    )

    random_schedule = make_schedule_dataframe(
        random_result["best_solution"],
        artists_df,
        config,
    )

    greedy_schedule = make_schedule_dataframe(
        greedy_result["best_solution"],
        artists_df,
        config,
    )

    random_schedule.to_csv(
        "results/best_schedule_random.csv",
        index=False,
    )

    greedy_schedule.to_csv(
        "results/best_schedule_greedy.csv",
        index=False,
    )

    print("\nBaseline results:")
    print(results_df)

    return results_df


if __name__ == "__main__":
    run_baselines(random_seed=42)