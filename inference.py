"""Run inference with a trained FIFA World Cup prediction model."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

REQUIRED_FILES = [
    "fifa_wc_model.keras",
    "scaler.pkl",
    "label_encoders.pkl",
    "team_elo.pkl",
    "team_history.pkl",
    "model_metadata.pkl",
]

QUALIFIED_TEAMS_2026 = [
    "United States", "Canada", "Mexico",
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
    "France", "Germany", "Spain", "England", "Portugal", "Netherlands", "Belgium",
    "Croatia", "Switzerland", "Austria", "Scotland", "Norway", "Denmark", "Poland", "Serbia",
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Qatar", "Jordan", "Uzbekistan",
    "Morocco", "Senegal", "Tunisia", "Algeria", "Egypt", "Ghana", "Cameroon", "Ivory Coast",
    "Costa Rica", "Panama", "Haiti", "New Zealand", "South Africa", "Curaçao", "Wales", "Cape Verde",
]


class FIFAWorldCupPredictor:
    """Loads saved artifacts and exposes prediction helpers."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._check_required_files()
        self._load_artifacts()

    def _check_required_files(self) -> None:
        missing = [f for f in REQUIRED_FILES if not (self.model_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing saved model files in {self.model_dir}: {', '.join(missing)}\n"
                "Run train.py first."
            )

    def _load_artifacts(self) -> None:
        self.model = keras.models.load_model(self.model_dir / "fifa_wc_model.keras")

        with open(self.model_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(self.model_dir / "label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)

        with open(self.model_dir / "team_elo.pkl", "rb") as f:
            self.final_team_elo = pickle.load(f)

        with open(self.model_dir / "team_history.pkl", "rb") as f:
            self.final_team_history = pickle.load(f)

        with open(self.model_dir / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.encoded_feature_columns = metadata["encoded_feature_columns"]
        self.default_elo = metadata["DEFAULT_ELO"]

    @staticmethod
    def get_tournament_weight(tournament_name) -> int:
        if tournament_name is None or (isinstance(tournament_name, float) and np.isnan(tournament_name)):
            return 1
        name = str(tournament_name).lower()
        if name == "fifa world cup":
            return 5
        if "qualification" in name or "qualifier" in name:
            return 3
        if "friendly" in name:
            return 1
        return 2

    def get_win_rate(self, team_name: str) -> float:
        last_30 = self.final_team_history.get(team_name, [])[-30:]
        return float(np.mean(last_30)) if last_30 else 0.5

    def encode_category(self, value, col_name: str) -> float:
        le = self.label_encoders[col_name]
        value = str(value)
        if value in le.classes_:
            return float(le.transform([value])[0])
        return float(len(le.classes_))

    def build_match_features(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
    ):
        home_elo = self.final_team_elo.get(team_a, self.default_elo)
        away_elo = self.final_team_elo.get(team_b, self.default_elo)

        feature_row = {
            "home_team_elo": home_elo,
            "away_team_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "home_team_win_rate": self.get_win_rate(team_a),
            "away_team_win_rate": self.get_win_rate(team_b),
            "is_neutral_venue": 1 if neutral else 0,
            "tournament_weight": self.get_tournament_weight(tournament),
            "home_team_encoded": self.encode_category(team_a, "home_team"),
            "away_team_encoded": self.encode_category(team_b, "away_team"),
            "tournament_encoded": self.encode_category(tournament, "tournament"),
            "city_encoded": self.encode_category("Unknown", "city"),
            "country_encoded": self.encode_category("Unknown", "country"),
        }

        X_one = np.array(
            [[feature_row[col] for col in self.encoded_feature_columns]],
            dtype=float,
        )
        return self.scaler.transform(X_one)

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        verbose: bool = True,
    ) -> tuple[float, float]:
        X_scaled = self.build_match_features(team_a, team_b, neutral=neutral, tournament=tournament)
        prob_a = float(self.model.predict(X_scaled, verbose=0)[0][0])
        prob_b = 1.0 - prob_a

        if verbose:
            favorite = team_a if prob_a >= prob_b else team_b
            print(f"Match: {team_a} vs {team_b}")
            print(f"  Tournament: {tournament} | Neutral venue: {neutral}")
            print(f"  {team_a} win probability: {prob_a:.2%}")
            print(f"  {team_b} win probability: {prob_b:.2%}")
            print(f"  Model favorite: {favorite}")

        return prob_a, prob_b

    def show_team_info(self, team_name: str) -> None:
        elo = self.final_team_elo.get(team_name, self.default_elo)
        win_rate = self.get_win_rate(team_name)
        num_matches = len(self.final_team_history.get(team_name, []))

        print(f"Team: {team_name}")
        print(f"  Elo rating: {elo:.1f}")
        print(f"  Recent win rate (last 30): {win_rate:.2%}")
        print(f"  Total matches in history: {num_matches}")
        if team_name not in self.final_team_elo:
            print(f"  Warning: team not found in saved Elo data — using default Elo {self.default_elo}.")

    def top_teams_by_elo(self, top_n: int = 15) -> pd.DataFrame:
        sorted_elo = sorted(self.final_team_elo.items(), key=lambda x: x[1], reverse=True)
        rows = [
            {"Rank": i + 1, "Team": team, "Elo": round(elo, 1)}
            for i, (team, elo) in enumerate(sorted_elo[:top_n])
        ]
        return pd.DataFrame(rows)

    def predict_batch(self, matchups: list[tuple[str, str]]) -> pd.DataFrame:
        rows = []
        for team_a, team_b in matchups:
            prob_a, prob_b = self.predict_match(team_a, team_b, verbose=False)
            rows.append({
                "Team A": team_a,
                "Team B": team_b,
                "A Win %": round(prob_a * 100, 1),
                "B Win %": round(prob_b * 100, 1),
                "Favorite": team_a if prob_a >= prob_b else team_b,
            })
        return pd.DataFrame(rows)

    def run_monte_carlo(self, num_simulations: int = 1000) -> list[tuple[str, float]]:
        assert len(QUALIFIED_TEAMS_2026) == 48

        def simulate_one_match(team_a: str, team_b: str) -> str:
            prob_a, _ = self.predict_match(team_a, team_b, verbose=False)
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
                key=lambda t: self.final_team_elo.get(t, self.default_elo),
                reverse=True,
            )
            knockout_teams += third_place_sorted[:8]
            return simulate_knockout(knockout_teams)

        win_counts: dict[str, int] = defaultdict(int)
        print(f"Running {num_simulations} Monte Carlo simulations...")
        for sim in range(num_simulations):
            champion = simulate_one_world_cup()
            win_counts[champion] += 1
            if (sim + 1) % 100 == 0:
                print(f"  Completed {sim + 1}/{num_simulations}...")

        win_probs = {team: count / num_simulations for team, count in win_counts.items()}
        return sorted(win_probs.items(), key=lambda x: x[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIFA World Cup model inference")
    parser.add_argument("--model-dir", type=Path, default=Path("."), help="Directory with saved model files")

    parser.add_argument("--team-a", type=str, help="First team (treated as home in features)")
    parser.add_argument("--team-b", type=str, help="Second team")
    parser.add_argument("--neutral", action="store_true", default=True, help="Neutral venue (default: True)")
    parser.add_argument("--tournament", type=str, default="FIFA World Cup", help="Tournament name")

    parser.add_argument("--team-info", type=str, help="Show Elo and win rate for one team")
    parser.add_argument("--top-elo", type=int, help="Print top N teams by Elo rating")
    parser.add_argument("--simulate", action="store_true", help="Run Monte Carlo World Cup simulation")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--examples", action="store_true", help="Run example predictions")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FIFAWorldCupPredictor(args.model_dir)

    print("Model loaded successfully!")
    print(f"Teams with Elo ratings: {len(predictor.final_team_elo)}")
    print(f"Input features: {len(predictor.encoded_feature_columns)}")

    if args.team_a and args.team_b:
        predictor.predict_match(args.team_a, args.team_b, neutral=args.neutral, tournament=args.tournament)
        return

    if args.team_info:
        predictor.show_team_info(args.team_info)
        return

    if args.top_elo:
        table = predictor.top_teams_by_elo(args.top_elo)
        print(f"\nTop {args.top_elo} teams by Elo:")
        print(table.to_string(index=False))
        return

    if args.simulate:
        top_teams = predictor.run_monte_carlo(num_simulations=args.simulations)
        print("\nTop 10 most likely 2026 World Cup winners:")
        for rank, (team, prob) in enumerate(top_teams[:10], start=1):
            print(f"{rank:2d}. {team:<20} {prob:.2%}")
        return

    if args.examples or len(sys.argv) == 1:
        print("\n--- Example predictions ---")
        predictor.predict_match("Brazil", "Argentina")
        print()
        predictor.predict_match("France", "Germany")
        print()

        print("--- Batch predictions ---")
        batch = predictor.predict_batch([
            ("Spain", "England"),
            ("Japan", "South Korea"),
            ("Argentina", "France"),
        ])
        print(batch.to_string(index=False))
        return

    print("No action specified. Try --team-a Brazil --team-b Argentina or --simulate")


if __name__ == "__main__":
    main()
