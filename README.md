📌 Overview
This project builds a machine‑learning model to predict whether the home team will win an NBA game. It uses a comprehensive dataset of NBA game statistics and applies:

Advanced team-level feature engineering
Rolling statistics (form indicators)
Win streak metrics
Rest-day & fatigue modelling
Season performance indicators
Random Forest and XGBoost classifiers

The goal is to demonstrate a complete, industry-quality ML workflow suitable for a data science or sports analytics portfolio.

🏀 Dataset
The dataset used in this project comes from Kaggle:
“NBA Basketball Dataset”
https://www.kaggle.com/datasets/wyattowalsh/basketball
It includes:

Game-level boxscore stats
Home/away team identifiers
Shooting percentages
Rebounds, assists, turnovers
Final points scored
Season information
Matchups
And more

This dataset is transformed into a team-centric table to enable time-series feature engineering.

🔧 Feature Engineering
This is the core of the project.
Feature engineering significantly increases predictive performance.
Below are the engineered features included in the final model.

⭐ 1. Rolling Averages (Team Form)
Rolling averages for each stat:

stat_pts
stat_reb
stat_ast
stat_stl
stat_blk
stat_tov
stat_fg_pct
stat_fg3_pct
stat_ft_pct

For windows:

3 games
5 games
10 games

Example features:
stat_pts_roll_5
stat_reb_roll_10
stat_fg_pct_roll_3

These capture recent team form.

⭐ 2. Win Streak Features
Momentum features showing how well a team has performed recently:
win_streak_1
win_streak_3
win_streak_5
win_streak_10

These represent wins in the last N games (shifted to prevent leakage).

⭐ 3. Rest Days (Fatigue Metrics)
Fatigue plays a big role in NBA performance.
Added features:

rest_days — days since last game
is_b2b — back‑to‑back indicator
rest_3plus — 3+ rest days


⭐ 4. Season Performance Indicators
These reflect long-term team strength:
season_win_pct        # cumulative win percentage (shifted)
season_wins           # cumulative win count
season_game_number    # progression through the season


⭐ 5. Home vs Away Feature Merging
Team-level features are merged back into a game-level row, creating:

Home team features → home_stat_*, home_win_streak_*, home_rest_days
Away team features → away_stat_*, away_win_streak_*, away_rest_days
Difference features:

diff_pts
diff_reb
diff_ast
diff_fg_pct
diff_pts_roll_5
diff_win_streak_3
...

Difference features are extremely strong predictors.

🤖 Machine Learning Models
Two main models were trained:
✔ 1. Logistic Regression (Baseline)
Serves as a simple benchmark.
Typical accuracy: 60–65%

✔ 2. Random Forest Classifier
A strong tree‑based model for tabular sports data.
Typical accuracy: 70–80%
Outputs include:

Classification report
Confusion matrix
Feature importance plot


✔ 3. XGBoost (Optional, Recommended)
Often the strongest model for this dataset.
Typical accuracy: 75–85%

📈 Model Evaluation
The project evaluates models using:

Accuracy
Precision / Recall / F1
Confusion Matrix
ROC Curve
Feature Importances

Example (you can replace with your actual numbers):
Random Forest Accuracy: 0.78
XGBoost Accuracy: 0.82


📁 Project Structure
nba-win-prediction/
│── data/
│   └── raw/                # Original Kaggle CSVs
│
│── notebooks/
│   └── 01-data-load-and-clean.ipynb
│   └── 02-feature-engineering.ipynb
│   └── 03-model-training.ipynb
│
│── src/
│   └── features.py         # (Optional) reusable feature functions
│   └── model.py            # (Optional) training scripts
│
│── outputs/
│   └── charts/             # Confusion matrix, feature importance, ROC curve
│   └── metrics/            # Classification reports, accuracy scores
│
│── README.md               # Project documentation


🚀 How to Run the Project
1. Clone the repo:
Shellgit clone https://github.com/<your-username>/nba-win-prediction.gitcd nba-win-predictionShow more lines
2. Install Python dependencies:
Shellpip install -r requirements.txtShow more lines
3. Place Kaggle dataset CSVs into:
data/raw/

4. Run the notebooks:

01-data-load-and-clean.ipynb
02-feature-engineering.ipynb
03-model-training.ipynb

5. View charts & results in:
outputs/


🔮 Future Work
Here are potential project extensions:

Add ELO rating system for team strength
Include travel distance & timezone shifts
Add betting line features (Vegas spread, O/U)
Explore deep learning architectures
Build a Streamlit or Dash app for game predictions
Hyperparameter tuning with Optuna


🎉 Conclusion
This project showcases a complete end‑to‑end data science workflow:

Data cleaning
Feature engineering
Rolling time‑series analytics
ML model training
Evaluation
GitHub‑ready documentation