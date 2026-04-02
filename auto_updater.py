# scripts/auto_update.py
import os
import sys
import time
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Repo root & paths
# ------------------------------
ROOT = Path(os.getenv("GITHUB_WORKSPACE", ".")).resolve()
RAW_PATH         = ROOT / "data" / "raw" / "games_master.csv"
CLEAN_PATH       = ROOT / "data" / "clean" / "clean_games.csv"
TEAM_GAMES_PATH  = ROOT / "data" / "modelling" / "team_games.csv"
MODEL_DATA_PATH  = ROOT / "data" / "modelling" / "model_data.csv"
MODELS_DIR       = ROOT / "models"

for p in [RAW_PATH.parent, CLEAN_PATH.parent, TEAM_GAMES_PATH.parent, MODEL_DATA_PATH.parent, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1) Load existing raw master
# ------------------------------
if RAW_PATH.exists():
    master = pd.read_csv(RAW_PATH)
else:
    master = pd.DataFrame(columns=[
        "game_id","game_date","team_id_home","team_name_home",
        "team_id_away","team_name_away","pts_home","pts_away"
    ])
print(f"[INFO] Loaded existing master: {master.shape}")

# ------------------------------
# 2) Fetch last 7 days completed games
# ------------------------------
print("[INFO] Fetching new games from NBA schedule API...")

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataUpdateBot/1.0; +https://github.com/)",
    "Accept": "application/json",
}
def fetch_json(url, headers=None, retries=3, timeout=20):
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"[WARN] Request attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)

data = fetch_json(SCHEDULE_URL, headers=HEADERS)
games = data.get("leagueSchedule", {}).get("gameDates", [])

cutoff = pd.Timestamp.now(tz="UTC")

for gdate in games:
    date_str = gdate.get("gameDate")

    try:
        date = pd.to_datetime(date_str, utc=True)
    except Exception:
        continue

    # Ensure date is UTC-aware (defensive)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")
    else:
        date = date.tz_convert("UTC")

    # cutoff safety (only needed once, but harmless here)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    if date < cutoff:
        continue

    # ⬇️ process future games here
    new_rows.append(...)

    for g in gdate.get("games", []):
        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}
        home_score = home.get("score")
        away_score = away.get("score")

        # only completed games have both scores
        if home_score is None or away_score is None:
            continue

        new_rows.append({
            "game_id": g.get("gameId"),
            "game_date": date.tz_convert(None),   # store naive local timestamp
            "team_id_home": home.get("teamId"),
            "team_name_home": home.get("teamName"),
            "team_id_away": away.get("teamId"),
            "team_name_away": away.get("teamName"),
            "pts_home": home_score,
            "pts_away": away_score
        })

new_df = pd.DataFrame(new_rows)
print(f"[INFO] New completed games found: {new_df.shape}")

if new_df.empty:
    print("[INFO] No new games in the last 7 days. Exiting successfully.")
    sys.exit(0)  # Important: success, but nothing to do

# ------------------------------
# 3) Append & save RAW
# ------------------------------
master = pd.concat([master, new_df], ignore_index=True)\
           .drop_duplicates(subset=["game_id"], keep="last")
master.to_csv(RAW_PATH, index=False)
print(f"[INFO] Updated RAW saved: {RAW_PATH} -> {master.shape}")

# ------------------------------
# 4) Basic clean
# ------------------------------
clean = master.copy()
clean["game_date"] = pd.to_datetime(clean["game_date"], errors="coerce")
clean = clean.dropna(subset=["pts_home", "pts_away"])
clean["home_win"] = (pd.to_numeric(clean["pts_home"], errors="coerce") >
                     pd.to_numeric(clean["pts_away"], errors="coerce")).astype(int)
clean.to_csv(CLEAN_PATH, index=False)
print(f"[INFO] Updated CLEAN saved: {CLEAN_PATH} -> {clean.shape}")

# ------------------------------
# 5) Feature engineering (home/away -> team rows; rolling; rest; streaks)
# ------------------------------
games_df = clean.copy()
home_cols = [c for c in games_df.columns if c.endswith("_home")]
away_cols = [c for c in games_df.columns if c.endswith("_away")]

home_df = games_df[["game_id","team_id_home","team_name_home","game_date"] + home_cols].copy()
home_df["team_id"] = home_df["team_id_home"]
home_df["team_name"] = home_df["team_name_home"]
home_df["team_win"] = games_df["home_win"]
home_df["home_away_flag"] = "home"
home_df = home_df.drop(columns=["team_id_home","team_name_home"], errors="ignore")

away_df = games_df[["game_id","team_id_away","team_name_away","game_date"] + away_cols].copy()
away_df["team_id"] = away_df["team_id_away"]
away_df["team_name"] = away_df["team_name_away"]
away_df["team_win"] = 1 - games_df["home_win"]
away_df["home_away_flag"] = "away"
away_df = away_df.drop(columns=["team_id_away","team_name_away"], errors="ignore")

home_df = home_df.rename(columns={c: "stat_" + c.replace("_home","") for c in home_cols})
away_df = away_df.rename(columns={c: "stat_" + c.replace("_away","") for c in away_cols})

common = sorted(set(home_df.columns) & set(away_df.columns))
home_df = home_df[common]
away_df = away_df[common]

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(["team_id","game_date"]).reset_index(drop=True)

stat_cols = [c for c in team_games.columns if c.startswith("stat_")
             and pd.api.types.is_numeric_dtype(team_games[c])]
for w in [3,5,10]:
    for stat in stat_cols:
        team_games[f"{stat}_roll_{w}"] = (
            team_games.groupby("team_id")[stat]
            .rolling(w).mean().shift(1).reset_index(level=0, drop=True)
        )

team_games["prev_game"] = team_games.groupby("team_id")["game_date"].shift(1)
team_games["rest_days"] = (team_games["game_date"] - team_games["prev_game"]).dt.days
team_games["is_b2b"] = (team_games["rest_days"] <= 1).astype(int)
team_games["rest_3plus"] = (team_games["rest_days"] >= 3).astype(int)

team_games["season"] = pd.to_datetime(team_games["game_date"]).dt.year
team_games["season_win_pct"] = (
    team_games.groupby(["team_id","season"])["team_win"]
    .expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
)

team_games.to_csv(TEAM_GAMES_PATH, index=False)
print(f"[INFO] team_games.csv updated: {TEAM_GAMES_PATH} -> {team_games.shape}")

# ------------------------------
# 6) Build model_data (merge home & away; diffs)
# ------------------------------
home_feats = team_games[team_games["home_away_flag"]=="home"].add_prefix("home_")
away_feats = team_games[team_games["home_away_flag"]=="away"].add_prefix("away_")

model_data = clean[["game_id","home_win","team_id_home","team_id_away"]].copy()
model_data = (model_data
    .merge(home_feats, left_on="game_id", right_on="home_game_id", how="left")
    .merge(away_feats, left_on="game_id", right_on="away_game_id", how="left")
)

model_data = model_data.drop(columns=[
    "home_game_id","away_game_id",
    "team_id_home","team_id_away",
    "home_team_id","away_team_id"
], errors="ignore")

numeric_home = [c for c in model_data.columns if c.startswith("home_stat_")]
numeric_away = [c for c in model_data.columns if c.startswith("away_stat_")]
for hc in numeric_home:
    base = hc.replace("home_stat_","")
    ac = "away_stat_" + base
    if ac in numeric_away:
        model_data[f"diff_{base}"] = (
            pd.to_numeric(model_data[hc], errors="coerce")
            - pd.to_numeric(model_data[ac], errors="coerce")
        )

model_data = model_data.fillna(0)
model_data.to_csv(MODEL_DATA_PATH, index=False)
print(f"[INFO] model_data.csv updated: {MODEL_DATA_PATH} -> {model_data.shape}")

# ------------------------------
# 7) Retrain model
# ------------------------------
y = model_data["home_win"].astype(int)
X = model_data.select_dtypes("number").drop(columns=["home_win"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODELS_DIR / "random_forest.pkl")
joblib.dump(list(X.columns), MODELS_DIR / "model_features.pkl")
print(f"[INFO] Model retrained — accuracy: {acc:.4f}")
print("[INFO] Model saved.")

print("\nAUTO-UPDATE COMPLETE ✔")
