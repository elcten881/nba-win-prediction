import os
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta


# ==========================================================
# 1. LOAD EXISTING DATA
# ==========================================================

RAW_PATH = "./data/raw/games_master.csv"
CLEAN_PATH = "./data/clean/clean_games.csv"
TEAM_GAMES_PATH = "./data/modelling/team_games.csv"
MODEL_DATA_PATH = "./data/modelling/model_data.csv"

if os.path.exists(RAW_PATH):
    master = pd.read_csv(RAW_PATH)
else:
    master = pd.DataFrame()

print("Loaded existing master:", master.shape)


# ==========================================================
# 2. FETCH NEW RESULTS FROM NBA.COM (LAST 7 DAYS)
# ==========================================================

print("Fetching new games from NBA.com...")

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
data = requests.get(SCHEDULE_URL).json()
games = data['leagueSchedule']['gameDates']

# last 7 days
cutoff = datetime.today() - timedelta(days=7)

new_rows = []
for gdate in games:
    date_str = gdate['gameDate']
    date = pd.to_datetime(date_str)

    if date < cutoff:
        continue  # skip games older than 7 days

    for g in gdate['games']:
        if 'homeTeam' not in g or 'awayTeam' not in g:
            continue

        # get final scores only (completed games)
        home_score = g['homeTeam'].get('score')
        away_score = g['awayTeam'].get('score')

        if home_score is None or away_score is None:
            continue  # skip future games and in-progress games

        new_rows.append({
            "game_id": g["gameId"],
            "game_date": date,
            "team_id_home": g['homeTeam']['teamId'],
            "team_name_home": g['homeTeam']['teamName'],
            "team_id_away": g['awayTeam']['teamId'],
            "team_name_away": g['awayTeam']['teamName'],
            "pts_home": home_score,
            "pts_away": away_score
        })

new_df = pd.DataFrame(new_rows)
print("New completed games found:", new_df.shape)

if new_df.empty:
    print("No new games to update.")
    exit()


# ==========================================================
# 3. APPEND TO MASTER & SAVE RAW
# ==========================================================

master = pd.concat([master, new_df], ignore_index=True).drop_duplicates(subset=["game_id"])
os.makedirs("./data/raw", exist_ok=True)
master.to_csv(RAW_PATH, index=False)
print("Updated RAW saved:", master.shape)


# ==========================================================
# 4. RE-RUN NOTEBOOK 1 LOGIC: BASIC CLEAN
# ==========================================================

clean = master.copy()

# ensure datetime
clean['game_date'] = pd.to_datetime(clean['game_date'], errors='coerce')

clean = clean.dropna(subset=['pts_home', 'pts_away'])
clean['home_win'] = (clean['pts_home'] > clean['pts_away']).astype(int)

os.makedirs("./data/clean", exist_ok=True)
clean.to_csv(CLEAN_PATH, index=False)
print("Updated CLEAN saved:", clean.shape)


# ==========================================================
# 5. RE-RUN NOTEBOOK 2 LOGIC: FEATURE ENGINEERING
# ==========================================================

games = clean.copy()

home_cols = [c for c in games.columns if c.endswith("_home")]
away_cols = [c for c in games.columns if c.endswith("_away")]

# build home team rows
home_df = games[['game_id','team_id_home','team_name_home','game_date'] + home_cols].copy()
home_df['team_id'] = home_df['team_id_home']
home_df['team_name'] = home_df['team_name_home']
home_df['team_win'] = games['home_win']
home_df['home_away_flag'] = 'home'
home_df = home_df.drop(columns=['team_id_home','team_name_home'])

# build away team rows
away_df = games[['game_id','team_id_away','team_name_away','game_date'] + away_cols].copy()
away_df['team_id'] = away_df['team_id_away']
away_df['team_name'] = away_df['team_name_away']
away_df['team_win'] = 1 - games['home_win']
away_df['home_away_flag'] = 'away'
away_df = away_df.drop(columns=['team_id_away','team_name_away'])

# normalize stat names
home_df = home_df.rename(columns={c:"stat_" + c.replace("_home","") for c in home_cols})
away_df = away_df.rename(columns={c:"stat_" + c.replace("_away","") for c in away_cols})

common = sorted(set(home_df.columns) & set(away_df.columns))
home_df = home_df[common]
away_df = away_df[common]

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(['team_id','game_date']).reset_index(drop=True)


# ---- rolling features ----
stat_cols = [c for c in team_games.columns if c.startswith("stat_") and pd.api.types.is_numeric_dtype(team_games[c])]
windows = [3,5,10]

for stat in stat_cols:
    for w in windows:
        team_games[f"{stat}_roll_{w}"] = (
            team_games.groupby("team_id")[stat]
            .rolling(w).mean().shift(1).reset_index(level=0, drop=True)
        )

# ---- win streaks ----
for w in [1,3,5,10]:
    team_games[f"win_streak_{w}"] = (
        team_games.groupby("team_id")['team_win']
        .rolling(w).sum().shift(1).reset_index(level=0, drop=True)
    )

# ---- rest days ----
team_games['prev_game'] = team_games.groupby('team_id')['game_date'].shift(1)
team_games['rest_days'] = (team_games['game_date'] - team_games['prev_game']).dt.days
team_games['is_b2b'] = (team_games['rest_days'] <= 1).astype(int)
team_games['rest_3plus'] = (team_games['rest_days'] >= 3).astype(int)

# ---- season win pct ----
team_games['season'] = pd.to_datetime(team_games['game_date']).dt.year
team_games['season_win_pct'] = (
    team_games.groupby(['team_id','season'])['team_win']
    .expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
)

# save
os.makedirs("./data/modelling", exist_ok=True)
team_games.to_csv(TEAM_GAMES_PATH, index=False)
print("team_games.csv updated:", team_games.shape)


# ==========================================================
# 6. REBUILD MODEL_DATA (NOTEBOOK 2 FINAL)
# ==========================================================

home_feats = team_games[team_games['home_away_flag']=='home'].add_prefix("home_")
away_feats = team_games[team_games['home_away_flag']=='away'].add_prefix("away_")

model_data = clean[['game_id','home_win','team_id_home','team_id_away']]

model_data = (
    model_data.merge(home_feats, left_on="game_id", right_on="home_game_id", how="left")
              .merge(away_feats, left_on="game_id", right_on="away_game_id", how="left")
)

# drop redundant
model_data = model_data.drop(columns=[
    "home_game_id","away_game_id",
    "team_id_home","team_id_away",
    "home_team_id","away_team_id"
], errors='ignore')

# numeric diffs
numeric_home = [c for c in model_data.columns if c.startswith("home_stat_")]
numeric_away = [c for c in model_data.columns if c.startswith("away_stat_")]

for hc in numeric_home:
    base = hc.replace("home_stat_","")
    ac = "away_stat_" + base
    if ac in numeric_away:
        model_data[f"diff_{base}"] = (
            pd.to_numeric(model_data[hc], errors='coerce') -
            pd.to_numeric(model_data[ac], errors='coerce')
        )

model_data = model_data.fillna(0)

model_data.to_csv(MODEL_DATA_PATH, index=False)
print("model_data.csv updated:", model_data.shape)


# ==========================================================
# 7. RETRAIN MODEL (NOTEBOOK 3 LOGIC)
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

y = model_data['home_win']
X = model_data.select_dtypes('number').drop(columns=['home_win'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

os.makedirs("./models", exist_ok=True)
joblib.dump(model, "./models/random_forest.pkl")
joblib.dump(list(X.columns), "./models/model_features.pkl")

print("Model retrained — accuracy:", acc)
print("Model saved.")


# ==========================================================
# 8. DONE!
# ==========================================================

print("\nAUTO-UPDATE COMPLETE ✔")