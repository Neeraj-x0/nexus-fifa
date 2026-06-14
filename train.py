"""Train the FIFA World Cup match prediction model."""

from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

K_FACTOR = 32
DEFAULT_ELO = 1500.0

NUMERIC_FEATURES = [
    "home_team_elo",
    "away_team_elo",
    "elo_difference",
    "home_team_win_rate",
    "away_team_win_rate",
    "is_neutral_venue",
    "tournament_weight",
]

CATEGORICAL_FEATURES = ["home_team", "away_team", "tournament", "city", "country"]

QUALIFIED_TEAMS_2026 = [
    "United States", "Canada", "Mexico",
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
    "France", "Germany", "Spain", "England", "Portugal", "Netherlands", "Belgium",
    "Croatia", "Switzerland", "Austria", "Scotland", "Norway", "Denmark", "Poland", "Serbia",
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Qatar", "Jordan", "Uzbekistan",
    "Morocco", "Senegal", "Tunisia", "Algeria", "Egypt", "Ghana", "Cameroon", "Ivory Coast",
    "Costa Rica", "Panama", "Haiti", "New Zealand", "South Africa", "Curaçao", "Wales", "Cape Verde",
]


def get_tournament_weight(tournament_name) -> int:
    if pd.isna(tournament_name):
        return 1
    name = str(tournament_name).lower()
    if name == "fifa world cup":
        return 5
    if "qualification" in name or "qualifier" in name:
        return 3
    if "friendly" in name:
        return 1
    return 2


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_df = pd.read_csv(data_dir / "results.csv")
    shootouts_df = pd.read_csv(data_dir / "shootouts.csv")
    former_names_df = pd.read_csv(data_dir / "former_names.csv")
    goalscorers_df = pd.read_csv(data_dir / "goalscorers.csv")

    print("Loaded CSV files:")
    print(f"  results:       {results_df.shape}")
    print(f"  shootouts:     {shootouts_df.shape}")
    print(f"  former_names:  {former_names_df.shape}")
    print(f"  goalscorers:   {goalscorers_df.shape}")

    return results_df, shootouts_df, former_names_df, goalscorers_df


def build_name_map(former_names_df: pd.DataFrame) -> dict[str, str]:
    return {row["former"]: row["current"] for _, row in former_names_df.iterrows()}


def standardize_team_name(team_name, name_map: dict[str, str]):
    if pd.isna(team_name):
        return team_name
    return name_map.get(team_name, team_name)


def clean_matches(results_df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    matches_df = results_df.copy()
    matches_df["home_team"] = matches_df["home_team"].apply(lambda t: standardize_team_name(t, name_map))
    matches_df["away_team"] = matches_df["away_team"].apply(lambda t: standardize_team_name(t, name_map))
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    matches_df = matches_df.sort_values("date").reset_index(drop=True)

    for col in ["tournament", "city", "country"]:
        matches_df[col] = matches_df[col].fillna("Unknown")

    matches_df["home_score"] = matches_df["home_score"].fillna(0)
    matches_df["away_score"] = matches_df["away_score"].fillna(0)
    matches_df["is_neutral_venue"] = matches_df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)

    return matches_df


def merge_shootouts(matches_df: pd.DataFrame, shootouts_df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    shootouts_clean = shootouts_df.copy()
    shootouts_clean["date"] = pd.to_datetime(shootouts_clean["date"])
    shootouts_clean["home_team"] = shootouts_clean["home_team"].apply(lambda t: standardize_team_name(t, name_map))
    shootouts_clean["away_team"] = shootouts_clean["away_team"].apply(lambda t: standardize_team_name(t, name_map))
    shootouts_clean = shootouts_clean[["date", "home_team", "away_team", "winner"]]
    shootouts_clean = shootouts_clean.rename(columns={"winner": "shootout_winner"})

    matches_df = matches_df.merge(
        shootouts_clean,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    conditions = [
        matches_df["home_score"] > matches_df["away_score"],
        matches_df["home_score"] < matches_df["away_score"],
        matches_df["shootout_winner"].notna(),
    ]
    choices = [
        matches_df["home_team"],
        matches_df["away_team"],
        matches_df["shootout_winner"],
    ]
    matches_df["winner"] = np.select(conditions, choices, default=None)

    shootout_fixed = (
        (matches_df["home_score"] == matches_df["away_score"]) & matches_df["shootout_winner"].notna()
    )
    print(f"Draws resolved by shootout: {shootout_fixed.sum()}")

    return matches_df


def compute_elo_features(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    team_elo: dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    matches_df["home_team_elo"] = np.nan
    matches_df["away_team_elo"] = np.nan

    for idx, row in matches_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = team_elo[home_team]
        away_elo = team_elo[away_team]

        matches_df.at[idx, "home_team_elo"] = home_elo
        matches_df.at[idx, "away_team_elo"] = away_elo

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        expected_away = 1.0 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        team_elo[home_team] = home_elo + K_FACTOR * (actual_home - expected_home)
        team_elo[away_team] = away_elo + K_FACTOR * (actual_away - expected_away)

    matches_df["elo_difference"] = matches_df["home_team_elo"] - matches_df["away_team_elo"]
    return matches_df, dict(team_elo)



def compute_win_rate_features(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    team_history: dict[str, list[float]] = defaultdict(list)
    matches_df["home_team_win_rate"] = np.nan
    matches_df["away_team_win_rate"] = np.nan

    for idx, row in matches_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_last_30 = team_history[home_team][-30:]
        away_last_30 = team_history[away_team][-30:]
        matches_df.at[idx, "home_team_win_rate"] = np.mean(home_last_30) if home_last_30 else 0.5
        matches_df.at[idx, "away_team_win_rate"] = np.mean(away_last_30) if away_last_30 else 0.5

        if row["home_score"] > row["away_score"]:
            home_result, away_result = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        team_history[home_team].append(home_result)
        team_history[away_team].append(away_result)

    return matches_df, dict(team_history)


def prepare_model_dataframe(matches_df: pd.DataFrame) -> pd.DataFrame:
    matches_df["tournament_weight"] = matches_df["tournament"].apply(get_tournament_weight)
    model_df = matches_df[matches_df["winner"].notna()].copy()
    model_df["home_win"] = (model_df["winner"] == model_df["home_team"]).astype(int)
    print(f"Model rows: {len(model_df)} (dropped {len(matches_df) - len(model_df)} draws)")
    return model_df


def split_and_encode(model_df: pd.DataFrame):
    train_df = model_df[model_df["date"] < "2010-01-01"].copy()
    test_df = model_df[
        (model_df["date"] >= "2010-01-01") & (model_df["tournament"] == "FIFA World Cup")
    ].copy()

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows:     {len(test_df)}")

    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_df[col + "_encoded"] = le.fit_transform(train_df[col].astype(str))

        known_classes = set(le.classes_)
        test_encoded = []
        for value in test_df[col].astype(str):
            if value in known_classes:
                test_encoded.append(le.transform([value])[0])
            else:
                test_encoded.append(len(le.classes_))
        test_df[col + "_encoded"] = test_encoded
        label_encoders[col] = le

    encoded_feature_columns = NUMERIC_FEATURES + [c + "_encoded" for c in CATEGORICAL_FEATURES]

    X_train = train_df[encoded_feature_columns].values.astype(float)
    y_train = train_df["home_win"].values.astype(float)
    X_test = test_df[encoded_feature_columns].values.astype(float)
    y_test = test_df["home_win"].values.astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        scaler,
        label_encoders,
        encoded_feature_columns,
    )


def build_model(num_features: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


def evaluate_model(model, X_test_scaled, y_test) -> None:
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away Win (0)", "Home Win (1)"]))


def save_artifacts(
    output_dir: Path,
    model,
    scaler,
    label_encoders,
    final_team_elo,
    final_team_history,
    encoded_feature_columns,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "fifa_wc_model.keras")

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    with open(output_dir / "team_elo.pkl", "wb") as f:
        pickle.dump(final_team_elo, f)

    with open(output_dir / "team_history.pkl", "wb") as f:
        pickle.dump(final_team_history, f)

    metadata = {
        "encoded_feature_columns": encoded_feature_columns,
        "DEFAULT_ELO": DEFAULT_ELO,
        "K_FACTOR": K_FACTOR,
    }
    with open(output_dir / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved model artifacts to {output_dir.resolve()}")


def get_win_rate(team_name: str, history_dict: dict[str, list[float]]) -> float:
    last_30 = history_dict.get(team_name, [])[-30:]
    return float(np.mean(last_30)) if last_30 else 0.5


def encode_category(value, col_name: str, label_encoders) -> float:
    le = label_encoders[col_name]
    value = str(value)
    if value in le.classes_:
        return float(le.transform([value])[0])
    return float(len(le.classes_))


def build_match_features(
    team_a: str,
    team_b: str,
    final_team_elo: dict[str, float],
    final_team_history: dict[str, list[float]],
    label_encoders,
    encoded_feature_columns,
    scaler,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
):
    home_elo = final_team_elo.get(team_a, DEFAULT_ELO)
    away_elo = final_team_elo.get(team_b, DEFAULT_ELO)

    feature_row = {
        "home_team_elo": home_elo,
        "away_team_elo": away_elo,
        "elo_difference": home_elo - away_elo,
        "home_team_win_rate": get_win_rate(team_a, final_team_history),
        "away_team_win_rate": get_win_rate(team_b, final_team_history),
        "is_neutral_venue": 1 if neutral else 0,
        "tournament_weight": get_tournament_weight(tournament),
        "home_team_encoded": encode_category(team_a, "home_team", label_encoders),
        "away_team_encoded": encode_category(team_b, "away_team", label_encoders),
        "tournament_encoded": encode_category(tournament, "tournament", label_encoders),
        "city_encoded": encode_category("Unknown", "city", label_encoders),
        "country_encoded": encode_category("Unknown", "country", label_encoders),
    }

    X_one = np.array([[feature_row[col] for col in encoded_feature_columns]], dtype=float)
    return scaler.transform(X_one)


def predict_match_prob(
    model,
    team_a: str,
    team_b: str,
    final_team_elo,
    final_team_history,
    label_encoders,
    encoded_feature_columns,
    scaler,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> tuple[float, float]:
    X_scaled = build_match_features(
        team_a,
        team_b,
        final_team_elo,
        final_team_history,
        label_encoders,
        encoded_feature_columns,
        scaler,
        neutral=neutral,
        tournament=tournament,
    )
    prob_a = float(model.predict(X_scaled, verbose=0)[0][0])
    return prob_a, 1.0 - prob_a


def run_monte_carlo(
    model,
    final_team_elo,
    final_team_history,
    label_encoders,
    encoded_feature_columns,
    scaler,
    num_simulations: int = 1000,
) -> list[tuple[str, float]]:
    assert len(QUALIFIED_TEAMS_2026) == 48

    def simulate_one_match(team_a: str, team_b: str) -> str:
        prob_a, _ = predict_match_prob(
            model,
            team_a,
            team_b,
            final_team_elo,
            final_team_history,
            label_encoders,
            encoded_feature_columns,
            scaler,
        )
        return team_a if random.random() < prob_a else team_b

    def play_group_stage(teams_in_group: list[str]):
        points = {team: 0 for team in teams_in_group}
        for i in range(len(teams_in_group)):
            for j in range(i + 1, len(teams_in_group)):
                winner = simulate_one_match(teams_in_group[i], teams_in_group[j])
                points[winner] += 3
        return sorted(points.items(), key=lambda x: x[1], reverse=True)

    def simulate_knockout(teams: list[str]) -> str:
        remaining = list(teams)
        random.shuffle(remaining)
        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining), 2):
                if i + 1 >= len(remaining):
                    next_round.append(remaining[i])
                else:
                    next_round.append(simulate_one_match(remaining[i], remaining[i + 1]))
            remaining = next_round
        return remaining[0]

    def simulate_one_world_cup() -> str:
        teams = QUALIFIED_TEAMS_2026.copy()
        random.shuffle(teams)
        groups = [teams[i * 4:(i + 1) * 4] for i in range(12)]

        group_winners, group_runners_up, third_place = [], [], []
        for group in groups:
            ranking = play_group_stage(group)
            group_winners.append(ranking[0][0])
            group_runners_up.append(ranking[1][0])
            third_place.append(ranking[2][0])

        knockout_teams = group_winners + group_runners_up
        third_place_sorted = sorted(
            third_place,
            key=lambda t: final_team_elo.get(t, DEFAULT_ELO),
            reverse=True,
        )
        knockout_teams += third_place_sorted[:8]
        return simulate_knockout(knockout_teams)

    win_counts: dict[str, int] = defaultdict(int)
    print(f"\nRunning {num_simulations} Monte Carlo simulations...")
    for sim in range(num_simulations):
        champion = simulate_one_world_cup()
        win_counts[champion] += 1
        if (sim + 1) % 100 == 0:
            print(f"  Completed {sim + 1}/{num_simulations}...")

    win_probs = {team: count / num_simulations for team, count in win_counts.items()}
    return sorted(win_probs.items(), key=lambda x: x[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FIFA World Cup prediction model")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save model artifacts")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--plot", action="store_true", help="Save training curve plot to output directory")
    parser.add_argument("--simulate", action="store_true", help="Run Monte Carlo World Cup simulation after training")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of Monte Carlo simulations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_df, shootouts_df, former_names_df, _goalscorers_df = load_data(args.data_dir)
    name_map = build_name_map(former_names_df)

    matches_df = clean_matches(results_df, name_map)
    matches_df = merge_shootouts(matches_df, shootouts_df, name_map)
    matches_df, team_elo = compute_elo_features(matches_df)
    matches_df, team_history = compute_win_rate_features(matches_df)
    model_df = prepare_model_dataframe(matches_df)

    (
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        scaler,
        label_encoders,
        encoded_feature_columns,
    ) = split_and_encode(model_df)

    model = build_model(X_train_scaled.shape[1])
    model.summary()

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1,
    )

    if args.plot:
        plot_training_history(history, args.output_dir / "training_curves.png")

    evaluate_model(model, X_test_scaled, y_test)

    final_team_elo = dict(team_elo)
    final_team_history = {k: list(v) for k, v in team_history.items()}

    save_artifacts(
        args.output_dir,
        model,
        scaler,
        label_encoders,
        final_team_elo,
        final_team_history,
        encoded_feature_columns,
    )

    prob_a, prob_b = predict_match_prob(
        model,
        "Brazil",
        "Argentina",
        final_team_elo,
        final_team_history,
        label_encoders,
        encoded_feature_columns,
        scaler,
    )
    print(f"\nExample prediction: Brazil vs Argentina -> Brazil {prob_a:.2%}, Argentina {prob_b:.2%}")

    if args.simulate:
        top_teams = run_monte_carlo(
            model,
            final_team_elo,
            final_team_history,
            label_encoders,
            encoded_feature_columns,
            scaler,
            num_simulations=args.simulations,
        )
        print("\nTop 10 most likely 2026 World Cup winners:")
        for rank, (team, prob) in enumerate(top_teams[:10], start=1):
            print(f"{rank:2d}. {team:<20} {prob:.2%}")


if __name__ == "__main__":
    main()
