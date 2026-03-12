
NBA Win Prediction Project

Overview
This project builds a full pipeline for predicting NBA scores and win probabilities using team-level game logs from the official NBA Stats API. It includes automated data updating, cleaning, feature engineering, and model training.

Project Structure
nba-win-prediction/
│
├── data/
│   ├── raw/
│   ├── clean/
│   │   ├── clean_game_nba.csv
│   │   ├── clean_game_nba_updated.csv
│
├── notebooks/
│
├── src/
│   ├── update_recent_games.py
│   ├── data_cleaning.py
│   ├── model_training.py
│   ├── predict.py
│
├── README.md
└── requirements.txt

Requirements
Install dependencies using:
pip install -r requirements.txt

Includes libraries: nba_api, pandas, numpy, scikit-learn, matplotlib.

Automatic Data Updating
The script update_recent_games.py fetches the last 30 days of NBA team game logs using nba_api and appends them to your cleaned dataset.

Data Cleaning & Feature Engineering
Cleans raw logs, creates modeling features (home/away, rolling averages, point differentials, rest days, etc.), and prepares labeled data.

Model Training
Trains machine learning models on the cleaned dataset and evaluates predictive accuracy.

Predictions
predict.py loads the trained model and produces win probabilities and projected score margins for upcoming games.

Usage
1. Update data: python src/update_recent_games.py
2. Train model: python src/model_training.py
3. Make predictions: python src/predict.py
4. Explore data: jupyter notebook

Future Improvements
- Automated scheduled updates
- Add player-level modeling features
- Dashboard visualization
