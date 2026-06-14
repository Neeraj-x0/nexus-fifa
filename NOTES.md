# FIFA World Cup ML Project — Beginner Notes

Everything we used in this project, explained in plain English.
Read this alongside the notebooks or scripts when something feels confusing.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Python Basics We Used](#2-python-basics-we-used)
3. [Libraries We Used](#3-libraries-we-used)
4. [The Dataset](#4-the-dataset)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Concepts](#7-machine-learning-concepts)
8. [Neural Networks (Keras)](#8-neural-networks-keras)
9. [Training the Model](#9-training-the-model)
10. [Evaluating the Model](#10-evaluating-the-model)
11. [Saving and Loading the Model](#11-saving-and-loading-the-model)
12. [Inference (Making Predictions)](#12-inference-making-predictions)
13. [Monte Carlo Simulation](#13-monte-carlo-simulation)
14. [Project Files Cheat Sheet](#14-project-files-cheat-sheet)
15. [Glossary](#15-glossary)

---

## 1. What This Project Does

**Goal:** Predict who wins an international football match, then use those predictions to simulate the 2026 World Cup many times.

**The big picture:**

```
Raw CSV files
    ↓
Clean & fix the data
    ↓
Create useful numbers (features) from each match
    ↓
Train a neural network to learn patterns
    ↓
Save the trained model to disk
    ↓
Use the model to predict new matches (inference)
    ↓
Simulate the World Cup 1,000 times (Monte Carlo)
```

**Two phases:**

| Phase | What you run | When |
|-------|--------------|------|
| **Training** | `fifa_world_cup_prediction.ipynb` or `train.py` | Once (or when you want to retrain) |
| **Inference** | `fifa_world_cup_inference.ipynb` or `inference.py` | Anytime after training |

---

## 2. Python Basics We Used

You don't need to be a Python expert, but these ideas show up everywhere in the code.

### Variables

A **variable** is a name that stores a value.

```python
team_name = "Brazil"
score = 3
```

### Lists and Dictionaries

- **List** — ordered collection: `["Brazil", "Argentina", "France"]`
- **Dictionary** — key → value pairs: `{"Brazil": 1650, "Argentina": 1620}`

We use dictionaries heavily for Elo ratings (`team_elo`) and match history (`team_history`).

### Functions

A **function** is a reusable block of code.

```python
def greet(name):
    return f"Hello, {name}"
```

Examples in our project: `predict_match()`, `get_tournament_weight()`, `clean_matches()`.

### Loops

**For loops** repeat code for each item:

```python
for team in ["Brazil", "France"]:
    print(team)
```

We loop through ~49,000 matches to compute Elo and win rates.

### `if / elif / else`

Make decisions in code:

```python
if home_score > away_score:
    winner = home_team
else:
    winner = away_team
```

### Importing

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```

This loads code from libraries so we don't rewrite everything ourselves.

### `defaultdict`

A special dictionary that gives a default value when a key is missing.

```python
from collections import defaultdict
team_elo = defaultdict(lambda: 1500.0)  # new teams start at 1500
```

### `pickle`

Saves Python objects (scaler, encoders, Elo dict) to a `.pkl` file and loads them back later.

### `argparse`

Lets scripts accept command-line flags:

```bash
python train.py --epochs 50 --plot
python inference.py --team-a Brazil --team-b Argentina
```

---

## 3. Libraries We Used

### pandas (`pd`)

**What it is:** Works with tables of data (like Excel in code).

**What we use it for:**
- Read CSV files: `pd.read_csv("results.csv")`
- View data: `.head()`, `.shape`, `.columns`
- Clean data: `.fillna()`, `.apply()`, `.merge()`
- Filter rows: `df[df["date"] < "2010-01-01"]`

**Key object:** `DataFrame` — a table with rows and columns.

---

### numpy (`np`)

**What it is:** Fast math on arrays of numbers.

**What we use it for:**
- `np.mean()` — average (for win rates)
- `np.select()` — pick values based on conditions (for shootout winners)
- `np.array()` — build feature arrays for the model
- `np.nan` — represents missing numbers

---

### scikit-learn (`sklearn`)

**What it is:** Classic machine learning tools (not deep learning).

**What we use it for:**

| Tool | Purpose |
|------|---------|
| `LabelEncoder` | Turn text like `"Brazil"` into numbers like `5` |
| `StandardScaler` | Scale features so they're comparable (mean 0, std 1) |
| `accuracy_score` | How often predictions are correct |
| `confusion_matrix` | Table of correct vs wrong predictions |
| `classification_report` | Precision, recall, F1-score |

---

### TensorFlow / Keras

**What it is:** Deep learning library. **Keras** is the easy API on top of TensorFlow.

**What we use it for:**
- Build the neural network (`keras.Sequential`)
- Add layers (`Dense`, `Dropout`, `Input`)
- Train the model (`.fit()`)
- Save/load the model (`.save()`, `keras.models.load_model()`)
- Predict (`.predict()`)

---

### matplotlib (`plt`)

**What it is:** Plotting and charts.

**What we use it for:**
- Plot training loss and accuracy over epochs
- Save the chart as `training_curves.png`

---

## 4. The Dataset

From Kaggle: [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2024) by martj42.

### `results.csv` — Main file

Every international match since 1872.

| Column | Meaning |
|--------|---------|
| `date` | When the match was played |
| `home_team` | Team playing at home |
| `away_team` | Team playing away |
| `home_score` | Goals scored by home team |
| `away_score` | Goals scored by away team |
| `tournament` | e.g. "FIFA World Cup", "Friendly" |
| `city`, `country` | Where the match was played |
| `neutral` | TRUE if neither team had home advantage |

### `shootouts.csv`

Penalty shootout results when a knockout match ended in a draw.

We merge this into results so we know the **real winner** (not just a tied score).

### `former_names.csv`

Maps old country names to current names.

Example: `"Upper Volta"` → `"Burkina Faso"`

Without this, the same country would appear under different names and confuse the model.

### `goalscorers.csv`

Individual goal scorers per match. We **load and explore** it in Step 1 but don't use it as a feature yet.

---

## 5. Data Cleaning

**Cleaning** = fixing messy data before the model sees it.

### Why clean?

- Missing values break calculations
- Old team names split one country into two
- Penalty shootouts look like draws in `results.csv`
- Dates stored as text need to become real dates

### Steps we did

1. **Standardize team names** using `former_names.csv`
2. **Convert dates** with `pd.to_datetime()`
3. **Sort by date** (oldest first) — critical for Elo and win rates
4. **Fill missing values** — `"Unknown"` for text, `0` for scores
5. **Convert `neutral`** from `"TRUE"/"FALSE"` text to `1`/`0` numbers
6. **Merge shootouts** to fix penalty winners
7. **Create `winner` column** — home win, away win, or shootout winner

### Merge (joining two tables)

Like VLOOKUP in Excel. We join `results` and `shootouts` on:

- `date` + `home_team` + `away_team`

When scores are tied but a shootout exists, we use the shootout winner.

---

## 6. Feature Engineering

**Features** = the input numbers the model uses to make a prediction.

**Feature engineering** = creating those numbers from raw data.

Think of it as: *"What information would help you guess the winner?"*

### Features we built

| Feature | What it means | Example |
|---------|---------------|---------|
| `home_team_elo` | Home team's strength rating | 1850 |
| `away_team_elo` | Away team's strength rating | 1720 |
| `elo_difference` | home_elo − away_elo | +130 (home stronger) |
| `home_team_win_rate` | Home team's win rate in last 30 games | 0.67 (67%) |
| `away_team_win_rate` | Away team's win rate in last 30 games | 0.53 |
| `is_neutral_venue` | 1 = neutral, 0 = home advantage | 1 |
| `tournament_weight` | How important the competition is | 5 for World Cup |
| `home_team_encoded` | Team name as a number | 42 |
| `away_team_encoded` | Team name as a number | 17 |
| `tournament_encoded` | Tournament as a number | 3 |
| `city_encoded` | City as a number | 88 |
| `country_encoded` | Country as a number | 12 |

### Elo rating (simple explanation)

**Elo** estimates how strong a team is. Started in chess, works great for football too.

- Every team starts at **1500** (average)
- **Beat a stronger team** → your Elo jumps up
- **Lose to a weaker team** → your Elo drops
- **Draw** → both teams move slightly toward each other

**Formula idea:**

```
expected_score = 1 / (1 + 10^((opponent_elo - your_elo) / 400))
new_elo = old_elo + K × (actual_result - expected_score)
```

- `K = 32` in our project (how much one match matters)
- `actual_result`: 1 = win, 0.5 = draw, 0 = loss

**Important:** We store each team's Elo **before** each match — never use future information (no cheating).

### Win rate (last 30 matches)

**Recent form** — how well has the team played lately?

- Look at last 30 results: 1 = win, 0 = loss, 0.5 = draw
- `win_rate = average of those results`
- No history? Use **0.5** (neutral guess)

Again, we only use matches **before** the current one.

### Neutral venue

- `1` = World Cup style (no home advantage)
- `0` = one team is at home

### Tournament weight

Not all matches are equally meaningful:

| Tournament | Weight |
|------------|--------|
| FIFA World Cup (finals) | 5 |
| Qualifiers | 3 |
| Friendly | 1 |
| Other (Copa América, Euros, etc.) | 2 |

### Target (label)

What we want the model to **predict**:

- **1** = home team won
- **0** = away team won
- **Draws removed** — we only do win/loss (binary classification)

---

## 7. Machine Learning Concepts

### Supervised learning

We show the model **examples with answers**:

```
Features (Elo, win rate, ...)  →  Label (home won? yes/no)
```

The model learns the pattern from thousands of past matches.

### Binary classification

Two possible outcomes: home win (1) or away win (0).

Not predicting the exact score — just **who wins**.

### Train / test split

We split data so we can check if the model works on **unseen** matches:

| Set | Rule | Purpose |
|-----|------|---------|
| **Train** | All matches before 2010 | Teach the model |
| **Test** | World Cup matches 2010+ | Check real performance |

**Why this split?** World Cup matches are the hardest, most important test — exactly what we care about.

### Data leakage (avoid cheating)

**Never** use future information when building features for a past match.

Bad: using a team's 2022 Elo to predict a 2010 match.

Good: using only data from before that match was played.

We also fit `LabelEncoder` and `StandardScaler` on **training data only**, then apply to test.

### LabelEncoder

Neural networks need numbers, not text.

```
"Brazil"     → 5
"Argentina"  → 2
"France"     → 8
```

Each unique text value gets an integer. We save the encoder so inference uses the same mapping.

### StandardScaler

Puts all features on a similar scale:

- Subtract the mean
- Divide by standard deviation

**Why?** Elo (~1500–2000) would dominate `is_neutral_venue` (0 or 1) without scaling.

After scaling, all features have mean ≈ 0 and std ≈ 1.

### Overfitting

When the model **memorizes** training data instead of learning general patterns.

Signs: great on training, bad on test.

**We reduce it with:**
- Dropout layers (randomly turn off neurons during training)
- Validation split (hold out 20% of training to monitor progress)

---

## 8. Neural Networks (Keras)

### What is a neural network?

A stack of layers that learns patterns from data — inspired by how brains work (loosely).

Each layer transforms numbers and passes them to the next layer.

### Our architecture

```
Input (12 features)
    ↓
Dense(64) + ReLU      ← 64 neurons, learn complex patterns
    ↓
Dropout(0.3)          ← randomly drop 30% (prevent overfitting)
    ↓
Dense(32) + ReLU
    ↓
Dropout(0.2)
    ↓
Dense(16) + ReLU
    ↓
Dense(1) + Sigmoid    ← output: probability 0 to 1
```

### Layer types we used

| Layer | What it does |
|-------|--------------|
| **Input** | Defines how many features go in (12 in our case) |
| **Dense** | Fully connected — every neuron connects to the previous layer |
| **Dropout** | Randomly disables neurons during training |
| **ReLU** | Activation function — keeps positive values, zeros negatives |
| **Sigmoid** | Squashes output to between 0 and 1 (perfect for probabilities) |

### Compile settings

| Setting | Value | Meaning |
|---------|-------|---------|
| **Optimizer** | Adam | Smart way to update weights during training |
| **Loss** | binary_crossentropy | Standard loss for yes/no problems |
| **Metric** | accuracy | Percent of correct predictions |

### Key terms

| Term | Meaning |
|------|---------|
| **Epoch** | One full pass through all training data |
| **Batch size** | How many samples the model sees before updating weights (64) |
| **Validation split** | 20% of training held out to check progress during training |
| **Weights** | Numbers inside the network that get learned |
| **Probability** | Model output between 0 and 1 (e.g. 0.73 = 73% chance home wins) |

---

## 9. Training the Model

### Command

```bash
python train.py
python train.py --epochs 100 --plot --simulate
```

### What happens during `.fit()`

1. Model sees a batch of matches (features + labels)
2. Makes predictions
3. Compares to actual results (loss)
4. Adjusts weights slightly to improve
5. Repeats for all batches → one **epoch**
6. Repeats for 100 epochs

### Training curves

We plot **loss** and **accuracy** over epochs:

- **Loss going down** = model is learning
- **Train vs validation** diverging = possible overfitting
- Saved as `training_curves.png` with `--plot`

---

## 10. Evaluating the Model

### Accuracy

```
Accuracy = correct predictions / total predictions
```

Example: 0.65 = right 65% of the time on World Cup 2010+ test matches.

### Confusion matrix

A 2×2 table:

```
                 Predicted
                 Away   Home
Actual  Away      [TN]   [FP]
        Home      [FN]   [TP]
```

- **TN** = correctly predicted away win
- **TP** = correctly predicted home win
- **FP / FN** = mistakes

### Classification report

Gives **precision**, **recall**, and **F1** for each class.

| Metric | Plain English |
|--------|---------------|
| **Precision** | When model says "home win", how often is it right? |
| **Recall** | Of all actual home wins, how many did we catch? |
| **F1** | Balance between precision and recall |

---

## 11. Saving and Loading the Model

After training once, we save **6 files**:

| File | Contains |
|------|----------|
| `fifa_wc_model.keras` | The trained neural network |
| `scaler.pkl` | StandardScaler (how to scale features) |
| `label_encoders.pkl` | Text → number mappings |
| `team_elo.pkl` | Final Elo rating per team |
| `team_history.pkl` | Match history for win rates |
| `model_metadata.pkl` | Feature column names, constants |

**Why separate files?** The model only stores the network. Elo, history, and encoders are needed to build the input features for new predictions.

### Loading

```python
model = keras.models.load_model("fifa_wc_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

Inference notebook/script does this automatically.

---

## 12. Inference (Making Predictions)

**Inference** = using a trained model on new data it hasn't seen.

### `predict_match(team_a, team_b)`

1. Look up Elo and win rate for both teams
2. Build the 12-feature row
3. Encode team/tournament names
4. Scale with saved scaler
5. Run through neural network
6. Output: probability team_a wins (team_b = 1 − that)

### Commands

```bash
python inference.py
python inference.py --team-a Brazil --team-b Argentina
python inference.py --top-elo 15
python inference.py --team-info France
python inference.py --simulate
```

### Team names matter

Use exact names from the dataset:

- `"United States"` not `"USA"`
- `"South Korea"` not `"Korea Republic"`

Check `results.csv` or run `--top-elo` if unsure.

---

## 13. Monte Carlo Simulation

### What is Monte Carlo?

Run something **many times with randomness** and see what happens on average.

Named after the Monte Carlo casino — it's about probability and chance.

### How we use it

1. Take 48 teams for 2026 World Cup
2. Randomly assign to 12 groups of 4
3. Simulate every group match using `predict_match` probabilities
4. Advance top 2 per group + 8 best third-place teams (32 total)
5. Simulate knockout bracket until one champion
6. Repeat **1,000 times**
7. Count how often each team wins → **win probability**

Example output:

```
 1. Argentina           18.2%
 2. France              14.7%
 3. Brazil              12.1%
 ...
```

**Note:** This takes several minutes — thousands of model predictions per run.

### Randomness

Each simulated match:

```python
if random.random() < prob_team_a_wins:
    winner = team_a
else:
    winner = team_b
```

A team with 70% win probability wins roughly 70% of simulated games — not every time.

---

## 14. Project Files Cheat Sheet

| File | Role |
|------|------|
| `results.csv` etc. | Raw Kaggle data |
| `fifa_world_cup_prediction.ipynb` | Step-by-step training (beginner tutorial) |
| `fifa_world_cup_inference.ipynb` | Step-by-step inference |
| `train.py` | Training script (same logic, command line) |
| `inference.py` | Inference script (same logic, command line) |
| `requirements.txt` | Python packages to install |
| `README.md` | Setup and quick start |
| `NOTES.md` | This file — concept reference |
| `fifa_wc_model.keras` + `.pkl` files | Saved model (created after training) |

### Typical workflow

```bash
pip install -r requirements.txt
python train.py --plot
python inference.py --team-a Brazil --team-b Argentina
python inference.py --simulate
```

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Feature** | An input number describing a match (e.g. Elo, win rate) |
| **Label / Target** | What we predict (1 = home win, 0 = away win) |
| **Model** | The trained neural network |
| **Training** | Showing the model examples so it learns |
| **Inference** | Using the trained model to predict new matches |
| **Epoch** | One full pass through training data |
| **Batch** | Small group of samples processed together |
| **Loss** | How wrong the model is (lower = better) |
| **Accuracy** | Percent of correct predictions |
| **Overfitting** | Memorizing training data, failing on new data |
| **Scaler** | Normalizes feature values to similar ranges |
| **Encoder** | Converts text categories to numbers |
| **Elo** | Rating system for team strength |
| **Binary classification** | Predicting one of two classes (win/loss) |
| **Sigmoid** | Function that outputs a probability between 0 and 1 |
| **Dropout** | Regularization — randomly disable neurons while training |
| **Pickle** | Python format for saving objects to disk |
| **DataFrame** | pandas table (rows and columns) |
| **Merge** | Join two tables on shared columns |
| **Monte Carlo** | Repeated random simulation to estimate probabilities |

---

## Quick Mental Model

When you're stuck, remember:

1. **Data** → clean it, make it consistent
2. **Features** → turn each match into numbers the model understands
3. **Train** → model finds patterns in old matches
4. **Test** → check on World Cup matches it never saw during training
5. **Save** → store everything needed to predict later
6. **Predict** → build features for a new match, run through model, get probability
7. **Simulate** → repeat predictions with randomness to estimate tournament outcomes

Machine learning isn't magic — it's pattern matching on historical data. The model learns *"teams with higher Elo and better recent form tend to win"* from thousands of examples.

---

*These notes match the code in `fifa_world_cup_prediction.ipynb`, `fifa_world_cup_inference.ipynb`, `train.py`, and `inference.py`.*
