from objective import (
    load_objective_data,
    evaluate_solution,
    get_solution_length,
    make_schedule_dataframe,
)


def main():
    artists_df, similarity_df, config = load_objective_data()

    solution_length = get_solution_length(config)

    solution = artists_df["artist_id"].sample(
        solution_length,
        random_state=42
    ).tolist()

    result = evaluate_solution(
        solution,
        artists_df,
        similarity_df,
        config,
    )

    print("Solution:")
    print(solution)

    print("\nScore:")
    print(result["score"])

    print("\nComponents:")
    print(result["components"])

    schedule_df = make_schedule_dataframe(solution, artists_df, config)

    print("\nSchedule:")
    print(schedule_df)


if __name__ == "__main__":
    main()