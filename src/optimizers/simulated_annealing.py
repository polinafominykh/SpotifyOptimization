import math
import time
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

try:
    from src.objective import (
        load_objective_data,
        evaluate_solution,
        get_solution_length,
        make_schedule_dataframe,
    )
    from src.baselines import random_baseline
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))

    from src.objective import (
        load_objective_data,
        evaluate_solution,
        get_solution_length,
        make_schedule_dataframe,
    )
    from src.baselines import random_baseline


def get_all_artist_ids(artists_df: pd.DataFrame) -> List[int]:
    return artists_df["artist_id"].astype(int).tolist()


def generate_random_solution(
    artists_df: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> List[int]:
    """
    Генерирует случайное расписание без повторов.
    """
    solution_length = get_solution_length(config)
    artist_ids = get_all_artist_ids(artists_df)

    if len(artist_ids) < solution_length:
        raise ValueError(
            f"Not enough artists: {len(artist_ids)}. "
            f"Required: {solution_length}."
        )

    return rng.choice(
        artist_ids,
        size=solution_length,
        replace=False,
    ).astype(int).tolist()


def get_neighbor_solution(
    solution: List[int],
    artists_df: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> List[int]:
    """
    Создает соседнее решение для Simulated Annealing.

    Возможные операции:
    1. swap — поменять двух артистов местами;
    2. replace — заменить одного артиста на нового;
    3. move_headliner — попробовать переместить хедлайнера в вечерний слот.
    """
    new_solution = solution.copy()

    operation = rng.choice(
        ["swap", "replace", "move_headliner"],
        p=[0.45, 0.40, 0.15],
    )

    if operation == "swap":
        return swap_two_positions(new_solution, rng)

    if operation == "replace":
        return replace_one_artist(new_solution, artists_df, rng)

    if operation == "move_headliner":
        return move_headliner_to_evening_slot(new_solution, artists_df, config, rng)

    return new_solution


def swap_two_positions(
    solution: List[int],
    rng: np.random.Generator,
) -> List[int]:
    """
    Меняет местами двух артистов в расписании.
    """
    if len(solution) < 2:
        return solution

    i, j = rng.choice(len(solution), size=2, replace=False)
    solution[i], solution[j] = solution[j], solution[i]

    return solution


def replace_one_artist(
    solution: List[int],
    artists_df: pd.DataFrame,
    rng: np.random.Generator,
) -> List[int]:
    """
    Заменяет одного артиста на нового, которого еще нет в расписании.
    """
    all_artist_ids = set(get_all_artist_ids(artists_df))
    selected_ids = set(solution)

    available_ids = list(all_artist_ids - selected_ids)

    if not available_ids:
        return solution

    position = int(rng.integers(0, len(solution)))
    new_artist = int(rng.choice(available_ids))

    solution[position] = new_artist

    return solution


def move_headliner_to_evening_slot(
    solution: List[int],
    artists_df: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> List[int]:
    """
    Пытается переместить случайного хедлайнера в вечерний слот.

    Это помогает SA быстрее уменьшать headliner_penalty.
    """
    n_scenes = int(config["festival"]["n_scenes"])
    n_slots = int(config["festival"]["n_slots"])

    artists_indexed = artists_df.set_index("artist_id")

    evening_slots = list(range(max(0, n_slots - 2), n_slots))
    evening_positions = []

    for slot_id in evening_slots:
        for scene_id in range(n_scenes):
            evening_positions.append(slot_id * n_scenes + scene_id)

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

    if not non_evening_headliner_positions:
        return swap_two_positions(solution, rng)

    old_position = int(rng.choice(non_evening_headliner_positions))
    new_position = int(rng.choice(evening_positions))

    solution[old_position], solution[new_position] = (
        solution[new_position],
        solution[old_position],
    )

    return solution


def acceptance_probability(
    current_score: float,
    candidate_score: float,
    temperature: float,
) -> float:
    """
    Вероятность принять новое решение.

    Так как мы максимизируем score:
    - если candidate_score лучше, принимаем всегда;
    - если хуже, принимаем с вероятностью exp(delta / T).
    """
    delta = candidate_score - current_score

    if delta >= 0:
        return 1.0

    if temperature <= 1e-12:
        return 0.0

    return math.exp(delta / temperature)


def simulated_annealing(
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    config: dict,
    initial_solution: List[int] | None = None,
    initial_temperature: float | None = None,
    cooling_rate: float | None = None,
    max_iterations: int | None = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulated Annealing для оптимизации расписания фестиваля.

    Метод максимизирует result["score"] из objective.py.
    """
    start_time = time.time()

    rng = np.random.default_rng(random_seed)

    if initial_temperature is None:
        initial_temperature = float(config.get("sa", {}).get("initial_temperature", 5.0))

    if cooling_rate is None:
        cooling_rate = float(config.get("sa", {}).get("cooling_rate", 0.995))

    if max_iterations is None:
        max_iterations = int(config.get("experiments", {}).get("max_iterations", 1000))

    if initial_solution is None:
        # Честный старт SA: случайное решение без повторов.
        current_solution = generate_random_solution(
            artists_df=artists_df,
            config=config,
            rng=rng,
        )
    else:
        current_solution = initial_solution.copy()

    current_result = evaluate_solution(
        current_solution,
        artists_df,
        similarity_df,
        config,
    )

    current_score = current_result["score"]

    best_solution = current_solution.copy()
    best_score = current_score
    best_components = current_result["components"]

    temperature = initial_temperature

    history = []
    current_history = []

    for iteration in range(max_iterations):
        candidate_solution = get_neighbor_solution(
            current_solution,
            artists_df,
            config,
            rng,
        )

        candidate_result = evaluate_solution(
            candidate_solution,
            artists_df,
            similarity_df,
            config,
        )

        candidate_score = candidate_result["score"]

        probability = acceptance_probability(
            current_score,
            candidate_score,
            temperature,
        )

        if rng.random() < probability:
            current_solution = candidate_solution
            current_score = candidate_score
            current_result = candidate_result

        if current_score > best_score:
            best_solution = current_solution.copy()
            best_score = current_score
            best_components = current_result["components"]

        history.append(best_score)
        current_history.append(current_score)

        temperature *= cooling_rate

    runtime = time.time() - start_time

    return {
        "method": "simulated_annealing",
        "best_solution": best_solution,
        "best_score": best_score,
        "history": history,
        "current_history": current_history,
        "components": best_components,
        "runtime": runtime,
        "iterations": max_iterations,
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
    }


def run_simulated_annealing(random_seed: int = 42) -> Dict[str, Any]:
    """
    Запускает SA и сохраняет результаты.
    """
    artists_df, similarity_df, config = load_objective_data()

    result = simulated_annealing(
        artists_df,
        similarity_df,
        config,
        random_seed=random_seed,
    )

    result_row = {
        "method": result["method"],
        "best_score": result["best_score"],
        "runtime": result["runtime"],
        "iterations": result["iterations"],
        "initial_temperature": result["initial_temperature"],
        "cooling_rate": result["cooling_rate"],
    }

    result_row.update(result["components"])

    results_df = pd.DataFrame([result_row])

    results_df.to_csv(
        "results/sa_results.csv",
        index=False,
    )

    schedule_df = make_schedule_dataframe(
        result["best_solution"],
        artists_df,
        config,
    )

    schedule_df.to_csv(
        "results/best_schedule_sa.csv",
        index=False,
    )

    history_df = pd.DataFrame(
        {
            "iteration": np.arange(len(result["history"])),
            "best_score": result["history"],
            "current_score": result["current_history"],
        }
    )

    history_df.to_csv(
        "results/sa_history.csv",
        index=False,
    )

    print("\nSimulated Annealing result:")
    print(results_df)

    return result


if __name__ == "__main__":
    run_simulated_annealing(random_seed=42)