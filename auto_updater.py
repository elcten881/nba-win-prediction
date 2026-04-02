# auto_updater.py

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

for p in [
    RAW_PATH.parent,
    CLEAN_PATH.parent,
    TEAM_GAMES_PATH.parent,
    MODEL_DATA_PATH.parent,
    MODELS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1) Load existing RAW master
# ------------------------------
if RAW_PATH.exists():
    master = pd.read_csv(RAW_PATH)
else:
    master = pd.DataFrame(columns=[
        "game_id",
        "game_date",
        "team_id_home",
        "team_name_home",
        "team_id_away",
        "team_name_away",
        "pts_home",
        "pts_away"
    ])

print(f"[INFO] Loaded existing master: {master.shape}")

# ------------------------------
# 2) Fetch completed games
# ------------------------------
print("[INFO] Fetching new games from NBA schedule API...")

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataUpdateBot/1.0)",
    "Accept": "application/json",
}

def fetch_json(url, headers=None, retries=3, timeout=20):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            print(f"[WARN] Attempt {attempt}/{retries} failed: {e}")

            if attempt == retries:
                print("[ERROR] NBA schedule API unreachable after retries. Skipping update.")
                return None

            time.sleep(2 * attempt)


    data = fetch_json(SCHEDULE_URL, headers=HEADERS)

    if not data:
        print("[INFO] No data fetched. Exiting successfully.")
        sys.exit(0)

    games = data.get("leagueSchedule", {}).get("gameDates", [])

    cutoff = pd.Timestamp.now(tz="UTC")
    new_rows = []

for gdate in games:
    date_str = gdate.get("gameDate")

    try:
        date = pd.to_datetime(date_str, utc=True)
    except Exception:
        continue

    # Skip past dates (timezone-safe)
    if date.value < cutoff.value:
        continue

    for g in gdate.get("games", []):
        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}

        home_score = home.get("score")
        away_score = away.get("score")

        # Only completed games
        if home_score is None or away_score is None:
            continue

        new_rows.append({
            "game_id": g.get("gameId"),
            "game_date": date.tz_convert(None),  # store naive UTC
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
    print("[INFO] No new games found. Exiting successfully.")
    sys.exit(0)

# ------------------------------
# 3) Append & save RAW
# ------------------------------
master = (
    pd.concat([master, new_df], ignore_index=True)
      .drop_duplicates(subset=["game_id"], keep="last")
)

master.to_csv(RAW_PATH, index=False)
print(f"[INFO] RAW updated: {RAW_PATH} -> {master.shape}")

# ------------------------------
# 4) Clean data
# ------------------------------
clean = master.copy()
clean["game_date"] = pd.to_datetime(clean["game_date"], errors="coerce")

clean = clean.dropna(subset=["pts_home", "pts_away"])

clean["home_win"] = (
    pd.to_numeric(clean["pts_home"], errors="coerce")
    > pd.to_numeric(clean["pts_away"], errors="coerce")
).astype(int)

clean.to_csv(CLEAN_PATH, index=False)
print(f"[INFO] CLEAN updated: {CLEAN_PATH} -> {clean.shape}")

# ------------------------------
# 5) Team‑level features
# ------------------------------
games_df = clean.copy()

home_cols = [c for c in games_df.columns if c.endswith("_home")]
away_cols = [c for c in games_df.columns if c.endswith("_away")]

home_df = games_df[["game_id", "team_id_home", "team_name_home", "game_date"] + home_cols].copy()
home_df["team_id"] = home_df["team_id_home"]
home_df["team_name"] = home_df["team_name_home"]
home_df["team_win"] = games_df["home_win"]
home_df["home_away_flag"] = "home"
home_df = home_df.drop(columns=["team_id_home", "team_name_home"], errors="ignore")

away_df = games_df[["game_id", "team_id_away", "team_name_away", "game_date"] + away_cols].copy()
away_df["team_id"] = away_df["team_id_away"]
away_df["team_name"] = away_df["team_name_away"]
away_df["team_win"] = 1 - games_df["home_win"]
away_df["home_away_flag"] = "away"
away_df = away_df.drop(columns=["team_id_away", "team_name_away"], errors="ignore")

home_df = home_df.rename(columns={c: "stat_" + c.replace("_home", "") for c in home_cols})
away_df = away_df.rename(columns={c: "stat_" + c.replace("_away", "") for c in away_cols})

common = sorted(set(home_df.columns) & set(away_df.columns))
team_games = pd.concat(
    [home_df[common], away_df[common]],
    ignore_index=True
)

team_games = team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

stat_cols = [
    c for c in team_games.columns
    if c.startswith("stat_") and pd.api.types.is_numeric_dtype(team_games[c])
]

for w in [3, 5, 10]:
    for stat in stat_cols:
        team_games[f"{stat}_roll_{w}"] = (
            team_games.groupby("team_id")[stat]
            .rolling(w)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

