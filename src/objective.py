from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"

ARTISTS_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "artists_features.csv"
SIMILARITY_MATRIX_PATH = ROOT_DIR / "data" / "processed" / "similarity_matrix.csv"


DEFAULT_WEIGHTS = {
    "appeal": 1.0,
    "diversity": 0.4,
    "flow": 0.3,
    "conflict": 0.6,
    "budget_penalty": 10.0,
    "duplicate_penalty": 100.0,
    "headliner_penalty": 30.0,
    "invalid_artist_penalty": 100.0,
}


AUDIO_FLOW_FEATURES = [
    "avg_energy",
    "avg_danceability",
    "avg_valence",
    "avg_tempo",
]


REQUIRED_ARTIST_COLUMNS = [
    "artist_id",
    "artist_name",
    "main_genre",
    "appeal_score",
    "cost",
    "is_headliner",
    "avg_energy",
    "avg_danceability",
    "avg_valence",
    "avg_tempo",
]


def load_config() -> dict:
    """
    Loads config.yaml.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    if "objective_weights" not in config:
        config["objective_weights"] = DEFAULT_WEIGHTS.copy()
    else:
        for key, value in DEFAULT_WEIGHTS.items():
            config["objective_weights"].setdefault(key, value)

    return config


def load_objective_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Loads artists_features.csv, similarity_matrix.csv and config.yaml.
    """
    config = load_config()

    if not ARTISTS_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Artists file not found: {ARTISTS_FEATURES_PATH}")

    if not SIMILARITY_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {SIMILARITY_MATRIX_PATH}")

    artists_df = pd.read_csv(ARTISTS_FEATURES_PATH)
    similarity_df = pd.read_csv(SIMILARITY_MATRIX_PATH, index_col=0)

    artists_df["artist_id"] = artists_df["artist_id"].astype(int)
    similarity_df.index = similarity_df.index.astype(int)
    similarity_df.columns = similarity_df.columns.astype(int)

    check_artists_dataframe(artists_df)

    return artists_df, similarity_df, config


def check_artists_dataframe(artists_df: pd.DataFrame) -> None:
    """
    Checks that artists_features.csv contains all required columns.
    """
    missing_columns = [
        column for column in REQUIRED_ARTIST_COLUMNS
        if column not in artists_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"artists_features.csv does not contain required columns: {missing_columns}"
        )


def get_festival_params(config: dict) -> tuple[int, int, int]:
    """
    Returns n_scenes, n_slots and budget from config.
    """
    festival_config = config["festival"]

    n_scenes = int(festival_config["n_scenes"])
    n_slots = int(festival_config["n_slots"])
    budget = int(festival_config["budget"])

    return n_scenes, n_slots, budget


def get_solution_length(config: dict) -> int:
    """
    Number of artists in one schedule.
    """
    n_scenes, n_slots, _ = get_festival_params(config)
    return n_scenes * n_slots


def solution_to_schedule(solution: list[int], config: dict) -> np.ndarray:
    """
    Converts flat solution list to schedule matrix.

    Shape:
    rows = time slots
    columns = scenes

    Example:
    schedule[slot_id, scene_id] = artist_id
    """
    n_scenes, n_slots, _ = get_festival_params(config)
    expected_length = n_scenes * n_slots

    if len(solution) != expected_length:
        raise ValueError(
            f"Invalid solution length: {len(solution)}. "
            f"Expected: {expected_length}"
        )

    return np.array(solution).reshape(n_slots, n_scenes)


def prepare_audio_features(artists_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales audio features for flow calculation.
    """
    artists_indexed = artists_df.set_index("artist_id")

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(artists_indexed[AUDIO_FLOW_FEATURES])

    audio_df = pd.DataFrame(
        scaled_features,
        index=artists_indexed.index,
        columns=AUDIO_FLOW_FEATURES,
    )

    return audio_df


def get_valid_artist_ids(artists_df: pd.DataFrame) -> set[int]:
    return set(artists_df["artist_id"].astype(int).tolist())


def calculate_invalid_artist_penalty(
    solution: list[int],
    artists_df: pd.DataFrame,
) -> float:
    """
    Penalty for artist IDs that do not exist in artists_features.csv.
    """
    valid_ids = get_valid_artist_ids(artists_df)
    invalid_count = sum(artist_id not in valid_ids for artist_id in solution)

    return invalid_count / max(len(solution), 1)


def calculate_duplicate_penalty(solution: list[int]) -> float:
    """
    Penalty for repeated artists.
    """
    total_count = len(solution)
    unique_count = len(set(solution))

    return (total_count - unique_count) / max(total_count, 1)


def calculate_appeal(
    solution: list[int],
    artists_df: pd.DataFrame,
) -> float:
    """
    Mean appeal score of selected artists, normalized to [0, 1].
    """
    artists_indexed = artists_df.set_index("artist_id")
    valid_ids = [artist_id for artist_id in solution if artist_id in artists_indexed.index]

    if not valid_ids:
        return 0.0

    appeal = artists_indexed.loc[valid_ids, "appeal_score"].mean()

    return float(appeal / 100)


def calculate_genre_diversity(
    solution: list[int],
    artists_df: pd.DataFrame,
) -> float:
    """
    Genre diversity based on entropy.
    Returns normalized value in [0, 1].
    """
    artists_indexed = artists_df.set_index("artist_id")
    valid_ids = [artist_id for artist_id in solution if artist_id in artists_indexed.index]

    if not valid_ids:
        return 0.0

    genres = artists_indexed.loc[valid_ids, "main_genre"]
    genre_counts = genres.value_counts()
    probabilities = genre_counts / genre_counts.sum()

    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))

    max_possible_genres = min(
        len(valid_ids),
        artists_df["main_genre"].nunique(),
    )

    if max_possible_genres <= 1:
        return 0.0

    normalized_entropy = entropy / np.log(max_possible_genres)

    return float(normalized_entropy)


def calculate_flow(
    solution: list[int],
    artists_df: pd.DataFrame,
    config: dict,
) -> float:
    """
    Measures smoothness of transitions between neighboring slots on each stage.

    Higher flow is better.
    """
    schedule = solution_to_schedule(solution, config)
    audio_df = prepare_audio_features(artists_df)

    distances = []

    for scene_id in range(schedule.shape[1]):
        for slot_id in range(schedule.shape[0] - 1):
            artist_a = int(schedule[slot_id, scene_id])
            artist_b = int(schedule[slot_id + 1, scene_id])

            if artist_a not in audio_df.index or artist_b not in audio_df.index:
                continue

            vector_a = audio_df.loc[artist_a].values
            vector_b = audio_df.loc[artist_b].values

            distance = np.linalg.norm(vector_a - vector_b)
            distances.append(distance)

    if not distances:
        return 0.0

    max_distance = np.sqrt(len(AUDIO_FLOW_FEATURES))
    mean_distance = np.mean(distances)

    flow = 1 - (mean_distance / max_distance)
    flow = np.clip(flow, 0, 1)

    return float(flow)


def calculate_conflict(
    solution: list[int],
    similarity_df: pd.DataFrame,
    config: dict,
) -> float:
    """
    Measures conflict between similar artists performing at the same time.

    Higher conflict is worse.
    """
    schedule = solution_to_schedule(solution, config)

    conflicts = []

    for slot_id in range(schedule.shape[0]):
        artists_in_slot = [int(artist_id) for artist_id in schedule[slot_id, :]]

        for artist_a, artist_b in combinations(artists_in_slot, 2):
            if artist_a in similarity_df.index and artist_b in similarity_df.columns:
                conflicts.append(similarity_df.loc[artist_a, artist_b])

    if not conflicts:
        return 0.0

    return float(np.mean(conflicts))


def calculate_total_cost(
    solution: list[int],
    artists_df: pd.DataFrame,
) -> float:
    """
    Total cost of selected artists.
    Duplicate artists are counted multiple times.
    """
    artists_indexed = artists_df.set_index("artist_id")
    valid_ids = [artist_id for artist_id in solution if artist_id in artists_indexed.index]

    if not valid_ids:
        return 0.0

    return float(artists_indexed.loc[valid_ids, "cost"].sum())


def calculate_budget_penalty(
    solution: list[int],
    artists_df: pd.DataFrame,
    config: dict,
) -> float:
    """
    Penalty for exceeding festival budget.
    """
    _, _, budget = get_festival_params(config)
    total_cost = calculate_total_cost(solution, artists_df)

    return float(max(0, (total_cost - budget) / budget))


def get_evening_slots(config: dict) -> list[int]:
    """
    Returns evening slots for headliners.

    For 6 slots:
    evening slots are [4, 5].
    """
    _, n_slots, _ = get_festival_params(config)

    if n_slots <= 2:
        return [n_slots - 1]

    return list(range(max(0, n_slots - 2), n_slots))


def calculate_headliner_penalty(
    solution: list[int],
    artists_df: pd.DataFrame,
    config: dict,
) -> float:
    """
    Penalty for placing headliners outside evening slots.
    """
    schedule = solution_to_schedule(solution, config)
    artists_indexed = artists_df.set_index("artist_id")

    evening_slots = set(get_evening_slots(config))

    selected_headliners = 0
    misplaced_headliners = 0

    for slot_id in range(schedule.shape[0]):
        for scene_id in range(schedule.shape[1]):
            artist_id = int(schedule[slot_id, scene_id])

            if artist_id not in artists_indexed.index:
                continue

            is_headliner = int(artists_indexed.loc[artist_id, "is_headliner"])

            if is_headliner == 1:
                selected_headliners += 1

                if slot_id not in evening_slots:
                    misplaced_headliners += 1

    if selected_headliners == 0:
        return 0.0

    return float(misplaced_headliners / selected_headliners)


def evaluate_solution(
    solution: list[int],
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Main objective function.

    All optimization methods must maximize result["score"].
    """
    weights = config["objective_weights"]

    appeal = calculate_appeal(solution, artists_df)
    diversity = calculate_genre_diversity(solution, artists_df)
    flow = calculate_flow(solution, artists_df, config)
    conflict = calculate_conflict(solution, similarity_df, config)

    budget_penalty = calculate_budget_penalty(solution, artists_df, config)
    duplicate_penalty = calculate_duplicate_penalty(solution)
    headliner_penalty = calculate_headliner_penalty(solution, artists_df, config)
    invalid_artist_penalty = calculate_invalid_artist_penalty(solution, artists_df)

    score = (
        weights["appeal"] * appeal
        + weights["diversity"] * diversity
        + weights["flow"] * flow
        - weights["conflict"] * conflict
        - weights["budget_penalty"] * budget_penalty
        - weights["duplicate_penalty"] * duplicate_penalty
        - weights["headliner_penalty"] * headliner_penalty
        - weights["invalid_artist_penalty"] * invalid_artist_penalty
    )

    total_cost = calculate_total_cost(solution, artists_df)
    _, _, budget = get_festival_params(config)

    return {
        "score": float(score),
        "components": {
            "appeal": float(appeal),
            "diversity": float(diversity),
            "flow": float(flow),
            "conflict": float(conflict),
            "budget_penalty": float(budget_penalty),
            "duplicate_penalty": float(duplicate_penalty),
            "headliner_penalty": float(headliner_penalty),
            "invalid_artist_penalty": float(invalid_artist_penalty),
            "total_cost": float(total_cost),
            "budget": float(budget),
            "budget_usage": float(total_cost / budget),
        },
    }


def make_schedule_dataframe(
    solution: list[int],
    artists_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Converts solution to readable schedule dataframe.
    """
    schedule = solution_to_schedule(solution, config)
    artists_indexed = artists_df.set_index("artist_id")

    rows = []

    for slot_id in range(schedule.shape[0]):
        for scene_id in range(schedule.shape[1]):
            artist_id = int(schedule[slot_id, scene_id])

            if artist_id in artists_indexed.index:
                artist = artists_indexed.loc[artist_id]

                rows.append(
                    {
                        "slot_id": slot_id,
                        "scene_id": scene_id,
                        "artist_id": artist_id,
                        "artist_name": artist["artist_name"],
                        "main_genre": artist["main_genre"],
                        "appeal_score": artist["appeal_score"],
                        "cost": artist["cost"],
                        "is_headliner": artist["is_headliner"],
                    }
                )
            else:
                rows.append(
                    {
                        "slot_id": slot_id,
                        "scene_id": scene_id,
                        "artist_id": artist_id,
                        "artist_name": "INVALID_ARTIST",
                        "main_genre": "unknown",
                        "appeal_score": 0,
                        "cost": 0,
                        "is_headliner": 0,
                    }
                )

    return pd.DataFrame(rows)


def print_solution_report(
    solution: list[int],
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    config: dict,
) -> None:
    """
    Prints short report for debugging.
    """
    result = evaluate_solution(solution, artists_df, similarity_df, config)

    print("Solution score:", result["score"])
    print("Components:")

    for key, value in result["components"].items():
        print(f"  {key}: {value:.4f}")