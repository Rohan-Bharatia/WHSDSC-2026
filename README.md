# 🏒 WHSDSC-2026 📊

### WHL Analytics & Season Simulation Engine

This repository contains the full modeling and simulation engine built for the **2026 Wharton High School Data Science Competition**, representing **Cherry Hill High School East**.

The project builds a predictive model for hockey games using shift-level data, generates ELO ratings, and runs large-scale season simulations to estimate standings, scoring output, and championship probabilities.

## 🚀 What This Project Does

The pipeline performs the following steps:

1. **Loads raw shift data** into a structured SQLite database
2. **Builds matchup features** from line combinations
3. **Constructs dynamic ELO ratings** from game results
4. **Trains LightGBM goal models** for home and away teams
5. **Simulates full seasons** thousands of times
6. **Aggregates standings statistics** across simulations

The result is a probabilistic forecast of:

* Average rank
* Average points
* Goals for / against
* Championship probability

## ⚙️ Installation

1. Install [Python](https://www.python.org/) (v3.10+)

Confirm the installation with:

```sh
python --version
```

2. (Recommended) Create a virtual environment:

```sh
python -m venv .venv
```

Windows:

```sh
.venv\Scripts\activate
```

macOS / Linux:

```sh
source .venv/bin/activate
```

3. Install dependencies:

```sh
pip install -r requirements.txt
```

## ▶️ Running the Full Simulation Pipeline

Run:

```sh
python main.py --seasons 1000
```

If `--seasons` is omitted, the default is 1000 simulated seasons.

During execution, the program will:

* Rebuild the database
* Train goal models
* Build ELO ratings
* Run parallel season simulations
* Print aggregated standings

Example output:

```
  1. Team A           Avg Rank: 2.14  Avg Pts: 94.3  GF: 243.2  GA: 201.8  Title%: 31.4%
...
```

A full example output after 5000 simulated seasons is linked at [`output.txt`](./output.txt)

## 📓 Jupyter Notebook Version

A notebook version of the pipeline is available at:

```
notebook/WHL_Simulation.py
```

To run it:

```sh
jupyter notebook
```

Then open the notebook in your browser.

The notebook walks through:

* Data loading
* Feature construction
* Model training
* Simulation
* Standings aggregation

This is ideal for experimentation and visualization.

## 🧠 Modeling Approach

### 🎯 Goal Prediction

Two LightGBM regression models are trained:

* Home goals model
* Away goals model

Features include:

* Expected goals (xG)
* Time on ice
* Rate metrics (per-60 stats)
* ELO ratings
* ELO differential

### 📈 ELO Rating System

An internal ELO system updates team strength using aggregated game performance.

Ratings are used both:

* As model features
* As dynamic inputs during simulation

### 🎲 Season Simulation

Each simulated season:

* Uses trained goal models
* Simulates each matchup
* Updates standings
* Tracks overtime results
* Repeats across N iterations

Parallel processing is used for performance.

## 📊 Output Metrics

For each team, the engine computes:

* **Average Rank**
* **Average Points**
* **Average Goals For**
* **Average Goals Against**
* **Championship Probability**

These are computed across all simulated seasons.

## 🔬 Reproducibility

* Global seed is fixed (`np.random.seed(42)`)
* Each simulated season uses deterministic seed offsets
* Database rebuild ensures clean state

This guarantees reproducible results across runs.

## 📝 License

This project is licensed under the MIT License.
See the [`LICENSE`](./LICENSE) file for details.
