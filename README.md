# FIFA World Cup Winner Prediction

A beginner-friendly machine learning project that predicts international football match outcomes using historical match data, then simulates the 2026 FIFA World Cup with Monte Carlo methods.

Built with **pandas**, **numpy**, **scikit-learn**, **TensorFlow/Keras**, and **matplotlib**.

## Dataset

This project uses the [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2024) dataset by martj42 on Kaggle.

Place these four CSV files in the project root:

| File | Description |
|------|-------------|
| `results.csv` | Every international match result |
| `shootouts.csv` | Penalty shootout winners |
| `former_names.csv` | Historical team name changes |
| `goalscorers.csv` | Individual goal scorers |

## Project structure

```
ml/
├── results.csv                      # Kaggle data (you provide)
├── shootouts.csv
├── former_names.csv
├── goalscorers.csv
├── fifa_world_cup_prediction.ipynb  # Train the model (run once)
├── fifa_world_cup_inference.ipynb   # Run predictions (no retraining)
├── train.py                         # Training script (CLI)
├── inference.py                     # Inference script (CLI)
├── requirements.txt
├── README.md
│
│   # Created after training (save cell):
├── fifa_wc_model.keras
├── scaler.pkl
├── label_encoders.pkl
├── team_elo.pkl
├── team_history.pkl
└── model_metadata.pkl
```

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+ (Python 3.11 or 3.12 recommended for best TensorFlow compatibility).

### 3. Add the CSV files

Download the dataset from Kaggle and copy the four CSV files into this folder.

## Usage

### Step 1 — Train the model (once)

**Option A — Notebook:** Open and run **`fifa_world_cup_prediction.ipynb`** from top to bottom.

**Option B — Script:**

```bash
python train.py
python train.py --plot --simulate --epochs 100
```

This loads the CSVs, trains the model, evaluates on World Cup 2010+ matches, and saves artifacts to disk.

Training takes several minutes depending on your hardware (100 epochs by default).

### Step 2 — Run inference (anytime)

**Option A — Notebook:** Open and run **`fifa_world_cup_inference.ipynb`**.

**Option B — Script:**

```bash
# Example predictions (default)
python inference.py

# Single match
python inference.py --team-a Brazil --team-b Argentina

# Top teams by Elo
python inference.py --top-elo 15

# Monte Carlo World Cup simulation
python inference.py --simulate --simulations 1000
```

## Saved model files

After training, these files are needed for inference:

| File | Purpose |
|------|---------|
| `fifa_wc_model.keras` | Trained neural network |
| `scaler.pkl` | Feature scaling (StandardScaler) |
| `label_encoders.pkl` | Categorical text → number mappings |
| `team_elo.pkl` | Final Elo rating per team |
| `team_history.pkl` | Match history for win-rate features |
| `model_metadata.pkl` | Feature column names and constants |

Keep all six files together in the same folder as the inference notebook.

## Model overview

| Component | Detail |
|-----------|--------|
| **Features** | Elo ratings, Elo difference, last-30 win rates, neutral venue, tournament weight, encoded team/tournament/location |
| **Target** | 1 = home team win, 0 = away team win (draws excluded) |
| **Train set** | All matches before 2010 |
| **Test set** | FIFA World Cup matches from 2010 onward |
| **Architecture** | Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Dense(16) → Dense(1, sigmoid) |
| **Optimizer** | Adam, binary crossentropy, 100 epochs |

## Example prediction

In the inference notebook:

```python
predict_match("France", "Germany")
```

Output:

```
Match: France vs Germany
  Tournament: FIFA World Cup | Neutral venue: True
  France win probability: 52.3%
  Germany win probability: 47.7%
  Model favorite: France
```

## Team names

Use names exactly as they appear in the dataset, for example:

- `"United States"` (not `"USA"`)
- `"South Korea"` (not `"Korea Republic"`)
- `"Ivory Coast"` (not `"Côte d'Ivoire"`)

Check `results.csv` or run the Elo table in the inference notebook if unsure.

## Notes

- **Monte Carlo simulation** (Step 9 in inference) can take several minutes — it runs thousands of model predictions.
- Edit `QUALIFIED_TEAMS_2026` in either notebook if the official 48-team list changes.
- `goalscorers.csv` is explored during training but not used as a model feature in the current version.
- Predictions are based on historical patterns and are for learning purposes — not betting advice.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` for `.pkl` or `.keras` files | Run the training notebook save cell first |
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` |
| Team not found / default Elo used | Use exact team name from the dataset |
| Notebook kernel not found | Run `python -m ipykernel install --user --name=fifa-ml` |

## License

The code in this repository is for educational use. The Kaggle dataset has its own license — check the dataset page on Kaggle before redistributing the CSV files.
