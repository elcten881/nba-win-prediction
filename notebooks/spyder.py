import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import datetime

# From notebook 1
games = pd.read_csv('../data/raw/game.csv')

games.head()

games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce')

games = games.dropna(subset=['pts_home', 'pts_away'])

games['home_win'] = (games['pts_home'] > games['pts_away']).astype(int)
games['home_win'].value_counts()

games.info()
games.describe(include='all')

games.to_csv('../data/clean/clean_game.csv', index=False)

# From notebook 2

games = pd.read_csv('../data/clean/clean_game.csv')

# Ensure date column is datetime
games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce')

games.head()

home_cols = [c for c in games.columns if c.endswith('_home')]
away_cols = [c for c in games.columns if c.endswith('_away')]

home_df = games[['game_id','season_id','game_date'] + home_cols].copy()
home_df['team_id'] = games['team_id_home']
home_df['team_name'] = games['team_name_home']
home_df['team_win'] = games['home_win']
home_df['home_away_flag'] = 'home'

away_df = games[['game_id','season_id','game_date'] + away_cols].copy()
away_df['team_id'] = games['team_id_away']
away_df['team_name'] = games['team_name_away']
away_df['team_win'] = 1 - games['home_win']
away_df['home_away_flag'] = 'away'

home_df = home_df.rename(columns={c: "stat_" + c.replace('_home', '') for c in home_cols})
away_df = away_df.rename(columns={c: "stat_" + c.replace('_away', '') for c in away_cols})

common_cols = sorted(set(home_df.columns).intersection(set(away_df.columns)))
home_df = home_df[common_cols]
away_df = away_df[common_cols]

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(['team_id', 'game_date']).reset_index(drop=True)
team_games.head()

stat_cols = [
    c for c in team_games.columns
    if c.startswith('stat_') and pd.api.types.is_numeric_dtype(team_games[c])
]

windows = [3, 5, 10]

for stat in stat_cols:
    for w in windows:
        team_games[f'{stat}_roll_{w}'] = (
            team_games
            .groupby('team_id')[stat]
            .rolling(w)
            .mean()
    .shift(1)
    .reset_index(level=0, drop=True)
)

streak_windows = [1,3,5,10]

for w in streak_windows:
    team_games[f'win_streak_{w}'] = (
        team_games
        .groupby('team_id')['team_win']
        .rolling(w)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    
team_games['prev_game_date'] = team_games.groupby('team_id')['game_date'].shift(1)
team_games['rest_days'] = (team_games['game_date'] - team_games['prev_game_date']).dt.days
team_games['is_b2b'] = (team_games['rest_days'] <= 1).astype(int)
team_games['rest_3plus'] = (team_games['rest_days'] >= 3).astype(int)

team_games = team_games.sort_values(['team_id','season_id','game_date'])

team_games['season_win_pct'] = (
    team_games
    .groupby(['team_id','season_id'])['team_win']
    .expanding()
    .mean()
    .shift(1)
    .reset_index(level=[0,1], drop=True)
)

team_games['season_wins'] = (
    team_games
    .groupby(['team_id','season_id'])['team_win']
    .cumsum()
    .shift(1)
)

team_games['season_game_number'] = (
    team_games.groupby(['team_id','season_id']).cumcount()
)

home_feats = team_games[team_games['home_away_flag']=='home'].copy()
away_feats = team_games[team_games['home_away_flag']=='away'].copy()

home_feats = home_feats.add_prefix('home_')
away_feats = away_feats.add_prefix('away_')

model_data = games[['game_id','home_win','team_id_home','team_id_away']].copy()

model_data = model_data.merge(
    home_feats, left_on='game_id', right_on='home_game_id', how='left'
).merge(
    away_feats, left_on='game_id', right_on='away_game_id', how='left'
)
        
model_data = model_data.drop(columns=[
    'home_game_id','away_game_id',
    'team_id_home','team_id_away',
    'home_team_id','away_team_id'
], errors='ignore')

numeric_home_stats = [
    c for c in model_data.columns
    if c.startswith('home_stat_') and pd.api.types.is_numeric_dtype(model_data[c])
]

numeric_away_stats = [
    c for c in model_data.columns
    if c.startswith('away_stat_') and pd.api.types.is_numeric_dtype(model_data[c])
]


diff_cols = []

# Identify numeric-only home/away stat columns
numeric_home_stats = [
    c for c in model_data.columns
    if c.startswith('home_stat_') and pd.api.types.is_numeric_dtype(model_data[c])
]

numeric_away_stats = [
    c for c in model_data.columns
    if c.startswith('away_stat_') and pd.api.types.is_numeric_dtype(model_data[c])
]

# Compute differences safely
for home_col in numeric_home_stats:
    base = home_col.replace('home_stat_', '')
    away_col = 'away_stat_' + base

    if away_col in numeric_away_stats:
        diff_col = 'diff_' + base

        model_data[diff_col] = (
            pd.to_numeric(model_data[home_col], errors='coerce')
            - pd.to_numeric(model_data[away_col], errors='coerce')
        )
        
        diff_cols.append(diff_col)

# Fill NaNs created from coercion or missing values
model_data = model_data.fillna(0)

# Quick check
print("Created diff columns:", len(diff_cols))
print(diff_cols[:15])

model_data = model_data.fillna(0)

cols_to_drop = [
    c for c in model_data.columns
    if 'team_id' in c or 'team_name' in c or 'matchup' in c
]

model_data = model_data.drop(columns=cols_to_drop, errors='ignore')

cols_to_drop = [
    c for c in model_data.columns if 'team_id' in c and 'roll' in c
]

model_data = model_data.drop(columns=cols_to_drop, errors='ignore')

model_data.to_csv('../data/modelling/model_data.csv', index=False)

model_data.head(), model_data.shape

os.makedirs('../data/modelling', exist_ok=True)

team_games.to_csv('../data/modelling/team_games.csv', index=False)
print("team_games.csv saved!")

#From notebook 3

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

model_data = pd.read_csv('../data/modelling/model_data.csv')
model_data.head()
model_data.shape

y = model_data['home_win'].astype(int)

X = model_data.select_dtypes(include='number').drop(columns=['home_win'], errors='ignore')

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

log_pred = log_reg.predict(X_test)

print("LogReg Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

disp = ConfusionMatrixDisplay.from_estimator(
    rf, X_test, y_test,
    cmap='Blues',
    values_format='d'
)
plt.title("Confusion Matrix — Random Forest")
plt.show()

RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve — Random Forest")
plt.show()

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top25 = importances.head(25)

plt.figure(figsize=(8,12))
top25.plot(kind='barh')
plt.title("Top 25 Feature Importances — Random Forest")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

top25

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# Save RandomForest
joblib.dump(rf, '../models/random_forest.pkl')

# Save X column order (critical for future prediction)
joblib.dump(list(X.columns), '../models/model_features.pkl')

print("Models saved.")

#From notebook 4

rf = joblib.load('../models/random_forest.pkl')
feature_cols = joblib.load('../models/model_features.pkl')
len(feature_cols)

team_games = pd.read_csv('../data/modelling/team_games.csv')
team_games['game_date'] = pd.to_datetime(team_games['game_date'], errors='coerce')
team_games.head()

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

data = requests.get(SCHEDULE_URL).json()

games = data['leagueSchedule']['gameDates']

rows = []

for gdate in games:
    date = gdate['gameDate']
    for g in gdate['games']:
        home = g['homeTeam']['teamName']
        visitor = g['awayTeam']['teamName']

        rows.append({
            "Date": date,
            "Home": home,
            "Visitor": visitor
        })

schedule = pd.DataFrame(rows)
schedule['Date'] = pd.to_datetime(schedule['Date'], errors='coerce')
schedule.head()

today = pd.Timestamp.today().normalize()
upcoming = schedule[schedule['Date'] >= today].sort_values('Date').reset_index(drop=True)

upcoming.head(10)

def latest_team_features(team_name):
    """
    Returns the last row in team_games for the given team.
    These are the most recent engineered features.
    """
    df = team_games[team_games['team_name'] == team_name].copy()

    if df.empty:
        # if team not found (different naming), return an empty row
        return pd.DataFrame()

    df = df.sort_values('game_date')
    latest = df.tail(1).copy()

    # remove this flag so prefixing later is clean
    latest = latest.drop(columns=['home_away_flag'], errors='ignore')
    
    return latest

def build_game_features(visitor, home):
    """
    Build ML-ready feature row for Visitor @ Home.
    """
    # Get latest stats for each team
    home_latest = latest_team_features(home).add_prefix('home_')
    away_latest = latest_team_features(visitor).add_prefix('away_')
    
    # If either team is missing, return None
    if home_latest.empty or away_latest.empty:
        return None
    
    # Combine horizontally
    row = pd.concat([home_latest, away_latest], axis=1)

    # Compute diff features
    for c in feature_cols:
        if c.startswith("diff_"):
            base = c.replace("diff_", "")
            hc = "home_stat_" + base
            ac = "away_stat_" + base
            if hc in row.columns and ac in row.columns:
                row[c] = row[hc] - row[ac]
            else:
                row[c] = 0

    # Final alignment to feature order
    row = row.reindex(columns=feature_cols, fill_value=0)

    return row

pred_rows = []

for _, g in upcoming.iterrows():
    date = g['Date']
    home = g['Home']
    visitor = g['Visitor']

    features = build_game_features(visitor, home)

    if features is None:
        print(f"Skipping {visitor} @ {home}: missing team features.")
        continue

    proba = rf.predict_proba(features)[0][1]
    pred  = rf.predict(features)[0]

    pred_rows.append({
        "Date": date.date(),
        "Visitor": visitor,
        "Home": home,
        "Home_Win_Probability (%)": round(proba*100, 1),
        "Predicted_Winner": home if pred == 1 else visitor
    })

pred_df = pd.DataFrame(pred_rows)
pred_df

pred_df.to_csv('../outputs/predictions/upcoming_predictions.csv', index=False)
pred_df.head()