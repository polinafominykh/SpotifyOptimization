from pathlib import Path
import ast
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT_DIR / "config.yaml"

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "spotify_tracks.csv"
ARTISTS_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "artists_features.csv"
SIMILARITY_MATRIX_PATH = ROOT_DIR / "data" / "processed" / "similarity_matrix.csv"

PLOTS_DIR = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================

def load_config() -> dict:
    """
    Загружает config.yaml, если он есть.
    Если config.yaml нет или там чего-то не хватает, используются дефолтные значения.
    """
    default_config = {
        "festival": {
            "n_artists": 200,
            "n_scenes": 3,
            "n_slots": 6,
            "budget": 700000,
        },
        "appeal_score": {
            "popularity_weight": 0.45,
            "energy_weight": 0.25,
            "danceability_weight": 0.20,
            "tracks_count_weight": 0.10,
        },
        "cost": {
            "base_cost": 3000,
            "popularity_coef": 350,
            "appeal_coef": 150,
            "noise_std": 1500,
        },
        "experiments": {
            "random_seed": 42,
        },
    }

    if not CONFIG_PATH.exists():
        print("config.yaml не найден, используются дефолтные параметры.")
        return default_config

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file)

    if user_config is None:
        return default_config

    for section, values in default_config.items():
        if section not in user_config:
            user_config[section] = values
        else:
            for key, value in values.items():
                user_config[section].setdefault(key, value)

    return user_config


# =========================
# Data cleaning
# =========================

REQUIRED_COLUMNS = [
    "artists",
    "track_name",
    "popularity",
    "track_genre",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "duration_ms",
]

NUMERIC_FEATURES = [
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "duration_ms",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит названия колонок к нижнему регистру и убирает пробелы.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    unnamed_columns = [col for col in df.columns if col.startswith("unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """
    Проверяет, что в датасете есть все нужные колонки.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"\nВ датасете не хватает колонок: {missing_columns}\n"
            f"Доступные колонки: {list(df.columns)}\n\n"
            f"Проверь, тот ли Spotify dataset загружен."
        )


def parse_artists(value) -> list:
    """
    Приводит поле artists к списку артистов.

    Возможные форматы:
    1. 'Taylor Swift'
    2. 'Artist A; Artist B'
    3. "['Artist A', 'Artist B']"
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(artist).strip() for artist in parsed if str(artist).strip()]
        except Exception:
            pass

    if ";" in value:
        return [artist.strip() for artist in value.split(";") if artist.strip()]

    return [value]


def normalize_artist_name(name: str) -> str:
    """
    Минимальная нормализация имени артиста.
    """
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def load_raw_dataset() -> pd.DataFrame:
    """
    Загружает сырой Spotify dataset.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nФайл не найден: {RAW_DATA_PATH}\n"
            f"Проверь, что датасет лежит тут: data/raw/spotify_tracks.csv"
        )

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw dataset shape: {df.shape}")

    df = clean_column_names(df)
    check_required_columns(df)

    return df


def clean_tracks_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Чистит треки:
    - убирает дубли;
    - убирает пропуски;
    - приводит числовые признаки;
    - фильтрует неадекватные значения;
    - разворачивает нескольких артистов в отдельные строки.
    """
    df = df.copy()

    keep_columns = REQUIRED_COLUMNS.copy()

    if "track_id" in df.columns:
        keep_columns = ["track_id"] + keep_columns

    df = df[keep_columns]

    before_duplicates = len(df)

    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    else:
        df = df.drop_duplicates(subset=["artists", "track_name"])

    print(f"Removed duplicates: {before_duplicates - len(df)}")

    df = df.dropna(subset=["artists", "track_name", "track_genre", "popularity"])

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES)

    df = df[(df["popularity"] >= 0) & (df["popularity"] <= 100)]
    df = df[(df["duration_ms"] >= 30_000) & (df["duration_ms"] <= 900_000)]
    df = df[(df["tempo"] > 0) & (df["tempo"] <= 250)]

    audio_features_0_1 = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
    ]

    for column in audio_features_0_1:
        df = df[(df[column] >= 0) & (df[column] <= 1)]

    df["artist_list"] = df["artists"].apply(parse_artists)
    df = df.explode("artist_list")

    df = df.rename(columns={"artist_list": "artist_name"})
    df["artist_name"] = df["artist_name"].apply(normalize_artist_name)

    df = df[df["artist_name"].notna()]
    df = df[df["artist_name"] != ""]

    print(f"Clean tracks shape: {df.shape}")
    print(f"Unique artists after cleaning: {df['artist_name'].nunique()}")
    print(f"Unique genres after cleaning: {df['track_genre'].nunique()}")

    return df


# =========================
# Artist aggregation
# =========================

def get_main_genre(series: pd.Series) -> str:
    """
    Возвращает самый частый жанр артиста.
    """
    mode_values = series.mode()

    if len(mode_values) == 0:
        return "unknown"

    return str(mode_values.iloc[0])


def aggregate_tracks_to_artists(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует треки в таблицу артистов.
    """
    print("Aggregating tracks to artist level...")

    grouped = tracks_df.groupby("artist_name")

    artists_df = grouped.agg(
        tracks_count=("track_name", "count"),
        avg_popularity=("popularity", "mean"),
        max_popularity=("popularity", "max"),
        avg_danceability=("danceability", "mean"),
        avg_energy=("energy", "mean"),
        avg_valence=("valence", "mean"),
        avg_tempo=("tempo", "mean"),
        avg_acousticness=("acousticness", "mean"),
        avg_instrumentalness=("instrumentalness", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        main_genre=("track_genre", get_main_genre),
        genres_count=("track_genre", "nunique"),
    ).reset_index()

    before_filter = len(artists_df)

    # Убираем артистов с одним треком, потому что по ним признаки слишком шумные
    artists_df = artists_df[artists_df["tracks_count"] >= 2].copy()

    print(f"Removed artists with only 1 track: {before_filter - len(artists_df)}")

    artists_df = artists_df.sort_values(
        by=["avg_popularity", "tracks_count"],
        ascending=False,
    ).reset_index(drop=True)

    artists_df.insert(0, "artist_id", np.arange(len(artists_df)))

    print(f"Artists dataset shape: {artists_df.shape}")

    return artists_df


# =========================
# Festival features
# =========================

def add_appeal_score(artists_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Создает festival appeal score.

    appeal_score — это proxy-оценка привлекательности артиста для фестиваля.
    """
    print("Adding appeal_score...")

    df = artists_df.copy()

    scaler = MinMaxScaler()

    df["popularity_norm"] = scaler.fit_transform(df[["avg_popularity"]])
    df["energy_norm"] = scaler.fit_transform(df[["avg_energy"]])
    df["danceability_norm"] = scaler.fit_transform(df[["avg_danceability"]])
    df["tracks_count_norm"] = scaler.fit_transform(np.log1p(df[["tracks_count"]]))

    weights = config["appeal_score"]

    df["appeal_score"] = (
        weights["popularity_weight"] * df["popularity_norm"]
        + weights["energy_weight"] * df["energy_norm"]
        + weights["danceability_weight"] * df["danceability_norm"]
        + weights["tracks_count_weight"] * df["tracks_count_norm"]
    ) * 100

    return df


def add_cost_duration_headliner(artists_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Добавляет признаки для оптимизационной задачи:
    - cost;
    - is_headliner;
    - duration_minutes.
    """
    print("Adding cost, duration and headliner flags...")

    df = artists_df.copy()

    random_seed = config["experiments"]["random_seed"]
    cost_config = config["cost"]

    rng = np.random.default_rng(random_seed)

    noise = rng.normal(
        loc=0,
        scale=cost_config["noise_std"],
        size=len(df),
    )

    df["cost"] = (
        cost_config["base_cost"]
        + cost_config["popularity_coef"] * df["avg_popularity"]
        + cost_config["appeal_coef"] * df["appeal_score"]
        + noise
    )

    df["cost"] = df["cost"].clip(lower=cost_config["base_cost"])
    df["cost"] = df["cost"].round(0).astype(int)

    df["duration_minutes"] = 60

    return df


def select_candidate_artists(artists_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Оставляет топ-N артистов-кандидатов для оптимизации.
    """
    n_artists = config["festival"]["n_artists"]

    print(f"Selecting top {n_artists} candidate artists...")

    candidates_df = (
        artists_df
        .sort_values(by="appeal_score", ascending=False)
        .head(n_artists)
        .copy()
        .reset_index(drop=True)
    )

    candidates_df["artist_id"] = np.arange(len(candidates_df))

    headliner_threshold = candidates_df["appeal_score"].quantile(0.90)

    candidates_df["is_headliner"] = (
            candidates_df["appeal_score"] >= headliner_threshold
    ).astype(int)

    conditions = [
        candidates_df["is_headliner"] == 1,
        candidates_df["appeal_score"] >= candidates_df["appeal_score"].quantile(0.70),
        candidates_df["appeal_score"] >= candidates_df["appeal_score"].quantile(0.35),
    ]

    choices = [90, 75, 60]
    candidates_df["duration_minutes"] = np.select(conditions, choices, default=45)

    final_columns = [
        "artist_id",
        "artist_name",
        "main_genre",
        "tracks_count",
        "genres_count",
        "avg_popularity",
        "max_popularity",
        "avg_danceability",
        "avg_energy",
        "avg_valence",
        "avg_tempo",
        "avg_acousticness",
        "avg_instrumentalness",
        "avg_duration_ms",
        "appeal_score",
        "cost",
        "duration_minutes",
        "is_headliner",
    ]

    return candidates_df[final_columns]


# =========================
# Similarity matrix
# =========================

def build_similarity_matrix(artists_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит матрицу похожести артистов.

    Похожесть нужна, чтобы штрафовать ситуацию,
    когда похожие артисты стоят одновременно на разных сценах.
    """
    print("Building similarity matrix...")

    features = [
        "avg_danceability",
        "avg_energy",
        "avg_valence",
        "avg_tempo",
        "avg_acousticness",
        "avg_instrumentalness",
    ]

    feature_matrix = artists_df[features].copy()

    scaler = MinMaxScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    similarity = cosine_similarity(feature_matrix_scaled)

    genres = artists_df["main_genre"].astype(str).to_numpy()

    genre_bonus = (genres[:, None] == genres[None, :]).astype(float) * 0.15

    similarity = np.clip(similarity + genre_bonus, 0, 1)

    similarity_df = pd.DataFrame(
        similarity,
        index=artists_df["artist_id"],
        columns=artists_df["artist_id"],
    )

    print(f"Similarity matrix shape: {similarity_df.shape}")

    return similarity_df


# =========================
# EDA plots and summary
# =========================

def plot_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path, bins: int = 30) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_eda_plots(tracks_df: pd.DataFrame, artists_df: pd.DataFrame) -> None:
    """
    Сохраняет базовые EDA-графики.
    """
    print("Saving EDA plots...")

    plot_distribution(
        tracks_df,
        "popularity",
        "Track popularity distribution",
        PLOTS_DIR / "track_popularity_distribution.png",
    )

    plot_distribution(
        artists_df,
        "appeal_score",
        "Artist appeal score distribution",
        PLOTS_DIR / "appeal_distribution.png",
    )

    plot_distribution(
        artists_df,
        "cost",
        "Artist estimated cost distribution",
        PLOTS_DIR / "cost_distribution.png",
    )

    genre_counts = artists_df["main_genre"].value_counts().head(15)

    plt.figure(figsize=(10, 6))
    genre_counts.sort_values().plot(kind="barh")
    plt.title("Top genres among candidate artists")
    plt.xlabel("Artists count")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "genre_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(artists_df["avg_popularity"], artists_df["cost"], alpha=0.7)
    plt.title("Artist popularity vs estimated cost")
    plt.xlabel("Average popularity")
    plt.ylabel("Estimated cost")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "popularity_vs_cost.png", dpi=200)
    plt.close()


def save_data_summary(
    tracks_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> None:
    """
    Сохраняет текстовый отчет по данным.
    """
    summary_path = RESULTS_DIR / "data_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Spotify Festival Optimization — Data Summary\n")
        file.write("=" * 70 + "\n\n")

        file.write("Tracks dataset\n")
        file.write("-" * 70 + "\n")
        file.write(f"Clean tracks shape: {tracks_df.shape}\n")
        file.write(f"Unique artists in tracks: {tracks_df['artist_name'].nunique()}\n")
        file.write(f"Unique genres in tracks: {tracks_df['track_genre'].nunique()}\n\n")

        file.write("Artists candidates dataset\n")
        file.write("-" * 70 + "\n")
        file.write(f"Artists shape: {artists_df.shape}\n")
        file.write(f"Headliners count: {artists_df['is_headliner'].sum()}\n")
        file.write(f"Average appeal score: {artists_df['appeal_score'].mean():.2f}\n")
        file.write(f"Average cost: {artists_df['cost'].mean():.2f}\n")
        file.write(f"Total estimated cost: {artists_df['cost'].sum():.2f}\n\n")

        file.write("Similarity matrix\n")
        file.write("-" * 70 + "\n")
        file.write(f"Similarity matrix shape: {similarity_df.shape}\n\n")

        file.write("Top 20 artists by appeal score\n")
        file.write("-" * 70 + "\n")

        top_artists = artists_df[
            [
                "artist_id",
                "artist_name",
                "main_genre",
                "appeal_score",
                "cost",
                "is_headliner",
            ]
        ].head(20)

        file.write(top_artists.to_string(index=False))
        file.write("\n\n")

        file.write("Top genres\n")
        file.write("-" * 70 + "\n")
        file.write(str(artists_df["main_genre"].value_counts().head(20)))

    print(f"Data summary saved to: {summary_path}")


# =========================
# Main
# =========================

def main() -> None:
    print("=" * 70)
    print("Spotify Festival Optimization — Data Preprocessing")
    print("=" * 70)

    config = load_config()

    raw_df = load_raw_dataset()
    clean_tracks_df = clean_tracks_dataset(raw_df)

    artists_df = aggregate_tracks_to_artists(clean_tracks_df)
    artists_df = add_appeal_score(artists_df, config)
    artists_df = add_cost_duration_headliner(artists_df, config)

    candidates_df = select_candidate_artists(artists_df, config)
    similarity_df = build_similarity_matrix(candidates_df)

    print("Saving processed files...")

    candidates_df.to_csv(ARTISTS_FEATURES_PATH, index=False)
    similarity_df.to_csv(SIMILARITY_MATRIX_PATH)

    save_eda_plots(clean_tracks_df, candidates_df)
    save_data_summary(clean_tracks_df, candidates_df, similarity_df)

    print("\nDone!")
    print(f"Artists features saved to: {ARTISTS_FEATURES_PATH}")
    print(f"Similarity matrix saved to: {SIMILARITY_MATRIX_PATH}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Summary saved to: {RESULTS_DIR / 'data_summary.txt'}")


if __name__ == "__main__":
    main()