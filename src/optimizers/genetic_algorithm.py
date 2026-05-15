import random
from pathlib import Path
import sys

import pandas as pd

# Добавляем папку src в путь, чтобы Python видел objective.py
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from objective import (
    load_objective_data,
    evaluate_solution,
    get_solution_length,
    make_schedule_dataframe,
)


def create_random_solution(artist_ids, solution_length):
    """
    Создает случайное расписание без повторов артистов.
    """
    return random.sample(artist_ids, solution_length)


def evaluate_population(population, artists_df, similarity_df, config):
    """
    Считает score для каждого решения в популяции.
    """
    evaluated = []

    for solution in population:
        result = evaluate_solution(solution, artists_df, similarity_df, config)
        evaluated.append({
            "solution": solution,
            "score": result["score"],
            "components": result["components"],
        })

    evaluated.sort(key=lambda x: x["score"], reverse=True)

    return evaluated


def tournament_selection(evaluated_population, tournament_size=3):
    """
    Турнирный отбор:
    случайно выбираем несколько решений и берем лучшее из них.
    """
    competitors = random.sample(evaluated_population, tournament_size)
    competitors.sort(key=lambda x: x["score"], reverse=True)
    return competitors[0]["solution"]


def crossover(parent_1, parent_2, solution_length):
    """
    Crossover для расписаний.

    Берем часть артистов из первого родителя,
    потом дополняем артистами из второго родителя без повторов.
    """
    cut_point = random.randint(1, solution_length - 1)

    child = parent_1[:cut_point].copy()

    for artist_id in parent_2:
        if artist_id not in child:
            child.append(artist_id)

        if len(child) == solution_length:
            break

    return child


def mutate(solution, artist_ids, mutation_rate):
    """
    Мутация:
    иногда заменяем одного артиста в расписании на нового,
    которого еще нет в solution.
    """
    mutated = solution.copy()

    if random.random() < mutation_rate:
        position = random.randint(0, len(mutated) - 1)

        available_artists = list(set(artist_ids) - set(mutated))

        if available_artists:
            mutated[position] = random.choice(available_artists)

    return mutated


def run_genetic_algorithm(
    artists_df,
    similarity_df,
    config,
    population_size=None,
    generations=None,
    mutation_rate=None,
    crossover_rate=None,
):
    """
    Запускает Genetic Algorithm.

    Возвращает:
    - best_solution
    - best_score
    - history со значениями score по поколениям
    """
    random_seed = config["experiments"].get("random_seed", 42)
    random.seed(random_seed)

    artist_ids = artists_df["artist_id"].tolist()
    solution_length = get_solution_length(config)

    ga_config = config.get("ga", {})

    if population_size is None:
        population_size = ga_config.get("population_size", 80)

    if generations is None:
        generations = config["experiments"].get("max_iterations", 500)

    if mutation_rate is None:
        mutation_rate = ga_config.get("mutation_rate", 0.15)

    if crossover_rate is None:
        crossover_rate = ga_config.get("crossover_rate", 0.8)

    # 1. Начальная популяция
    population = [
        create_random_solution(artist_ids, solution_length)
        for _ in range(population_size)
    ]

    best_solution = None
    best_score = float("-inf")
    best_components = None

    history = []

    # 2. Основной цикл
    for generation in range(generations):
        evaluated_population = evaluate_population(
            population,
            artists_df,
            similarity_df,
            config,
        )

        current_best = evaluated_population[0]

        if current_best["score"] > best_score:
            best_score = current_best["score"]
            best_solution = current_best["solution"].copy()
            best_components = current_best["components"]

        history.append({
            "iteration": generation,
            "best_score": best_score,
            "current_best_score": current_best["score"],
        })

        # 3. Elitism: сохраняем несколько лучших решений
        elite_count = max(1, population_size // 10)
        new_population = [
            item["solution"] for item in evaluated_population[:elite_count]
        ]

        # 4. Создаем новое поколение
        while len(new_population) < population_size:
            parent_1 = tournament_selection(evaluated_population)
            parent_2 = tournament_selection(evaluated_population)

            if random.random() < crossover_rate:
                child = crossover(parent_1, parent_2, solution_length)
            else:
                child = parent_1.copy()

            child = mutate(child, artist_ids, mutation_rate)

            new_population.append(child)

        population = new_population

    return {
        "method": "Genetic Algorithm",
        "best_solution": best_solution,
        "best_score": best_score,
        "best_components": best_components,
        "history": pd.DataFrame(history),
    }


def main():
    artists_df, similarity_df, config = load_objective_data()

    result = run_genetic_algorithm(
        artists_df,
        similarity_df,
        config,
    )

    print("Best score:")
    print(result["best_score"])

    print("\nBest components:")
    print(result["best_components"])

    print("\nBest solution:")
    print(result["best_solution"])

    schedule_df = make_schedule_dataframe(
        result["best_solution"],
        artists_df,
        config,
    )

    print("\nBest schedule:")
    print(schedule_df)

    # Сохраняем результат
    root_dir = Path(__file__).resolve().parents[2]
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    schedule_df.to_csv(
        results_dir / "best_schedule_ga.csv",
        index=False,
    )

    result["history"].to_csv(
        results_dir / "ga_history.csv",
        index=False,
    )

    print("\nSaved:")
    print(results_dir / "best_schedule_ga.csv")
    print(results_dir / "ga_history.csv")


if __name__ == "__main__":
    main()