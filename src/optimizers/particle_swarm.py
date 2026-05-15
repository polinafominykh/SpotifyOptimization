import random
from pathlib import Path
import sys

import pandas as pd

# Чтобы файл видел objective.py
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


def get_score(solution, artists_df, similarity_df, config):
    """
    Считает score через общую objective function.
    """
    result = evaluate_solution(solution, artists_df, similarity_df, config)
    return result["score"], result["components"]


def repair_solution(solution, artist_ids, solution_length):
    """
    Исправляет solution:
    - убирает повторы;
    - добавляет недостающих артистов;
    - обрезает до нужной длины.
    """
    repaired = []
    used = set()

    for artist_id in solution:
        if artist_id not in used and artist_id in artist_ids:
            repaired.append(artist_id)
            used.add(artist_id)

        if len(repaired) == solution_length:
            break

    available_artists = list(set(artist_ids) - used)

    while len(repaired) < solution_length:
        new_artist = random.choice(available_artists)
        repaired.append(new_artist)
        used.add(new_artist)
        available_artists.remove(new_artist)

    return repaired


def update_particle(
    current_solution,
    personal_best,
    global_best,
    artist_ids,
    solution_length,
    copy_probability,
    mutation_probability,
):
    """
    Обновляет частицу для дискретного PSO.

    В обычном PSO частица двигается по числовому пространству.
    У нас расписание дискретное, поэтому движение заменяем операциями:
    - часть позиций копируем из personal_best;
    - часть позиций копируем из global_best;
    - иногда делаем случайную замену артиста.
    """
    new_solution = current_solution.copy()

    for i in range(solution_length):
        random_value = random.random()

        if random_value < copy_probability / 2:
            new_solution[i] = personal_best[i]

        elif random_value < copy_probability:
            new_solution[i] = global_best[i]

    # Мутация: случайно заменяем одного артиста
    if random.random() < mutation_probability:
        position = random.randint(0, solution_length - 1)

        available_artists = list(set(artist_ids) - set(new_solution))

        if available_artists:
            new_solution[position] = random.choice(available_artists)

    new_solution = repair_solution(
        new_solution,
        artist_ids,
        solution_length,
    )

    return new_solution


def run_particle_swarm(
    artists_df,
    similarity_df,
    config,
    n_particles=None,
    iterations=None,
    copy_probability=None,
    mutation_probability=None,
):
    """
    Запускает Discrete Particle Swarm Optimization.

    Возвращает:
    - best_solution
    - best_score
    - history
    """
    random_seed = config["experiments"].get("random_seed", 42)
    random.seed(random_seed)

    artist_ids = artists_df["artist_id"].tolist()
    solution_length = get_solution_length(config)

    pso_config = config.get("pso", {})

    if n_particles is None:
        n_particles = pso_config.get("n_particles", 50)

    if iterations is None:
        iterations = config["experiments"].get("max_iterations", 500)

    if copy_probability is None:
        copy_probability = pso_config.get("copy_probability", 0.4)

    if mutation_probability is None:
        mutation_probability = pso_config.get("mutation_probability", 0.2)

    # 1. Создаем начальный рой
    particles = [
        create_random_solution(artist_ids, solution_length)
        for _ in range(n_particles)
    ]

    # 2. Personal best для каждой частицы
    personal_best_solutions = []
    personal_best_scores = []
    personal_best_components = []

    for particle in particles:
        score, components = get_score(
            particle,
            artists_df,
            similarity_df,
            config,
        )

        personal_best_solutions.append(particle.copy())
        personal_best_scores.append(score)
        personal_best_components.append(components)

    # 3. Global best
    best_index = personal_best_scores.index(max(personal_best_scores))

    global_best_solution = personal_best_solutions[best_index].copy()
    global_best_score = personal_best_scores[best_index]
    global_best_components = personal_best_components[best_index]

    history = []

    # 4. Основной цикл PSO
    for iteration in range(iterations):
        for i in range(n_particles):
            new_particle = update_particle(
                current_solution=particles[i],
                personal_best=personal_best_solutions[i],
                global_best=global_best_solution,
                artist_ids=artist_ids,
                solution_length=solution_length,
                copy_probability=copy_probability,
                mutation_probability=mutation_probability,
            )

            new_score, new_components = get_score(
                new_particle,
                artists_df,
                similarity_df,
                config,
            )

            particles[i] = new_particle

            # Обновляем personal best
            if new_score > personal_best_scores[i]:
                personal_best_scores[i] = new_score
                personal_best_solutions[i] = new_particle.copy()
                personal_best_components[i] = new_components

            # Обновляем global best
            if new_score > global_best_score:
                global_best_score = new_score
                global_best_solution = new_particle.copy()
                global_best_components = new_components

        history.append({
            "iteration": iteration,
            "best_score": global_best_score,
        })

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: best_score = {global_best_score:.4f}")

    return {
        "method": "Discrete PSO",
        "best_solution": global_best_solution,
        "best_score": global_best_score,
        "best_components": global_best_components,
        "history": pd.DataFrame(history),
    }


def main():
    artists_df, similarity_df, config = load_objective_data()

    # Быстрый тестовый запуск
    result = run_particle_swarm(
        artists_df,
        similarity_df,
        config,
        n_particles=20,
        iterations=30,
        copy_probability=0.4,
        mutation_probability=0.2,
    )

    print("\nBest score:")
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
        results_dir / "best_schedule_pso.csv",
        index=False,
    )

    result["history"].to_csv(
        results_dir / "pso_history.csv",
        index=False,
    )

    print("\nSaved:")
    print(results_dir / "best_schedule_pso.csv")
    print(results_dir / "pso_history.csv")


if __name__ == "__main__":
    main()