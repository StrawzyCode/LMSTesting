from __future__ import annotations
import pandas as pd

# Add/extend this mapping as needed based on what you see in your files
TEAM_ALIASES = {
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",

    "Man Utd": "Man United",
    "Manchester United": "Man United",

    "Man City": "Man City",
    "Manchester City": "Man City",

    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",

    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",

    "Wolves": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
}

def norm_team(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip()
    return TEAM_ALIASES.get(x, x)

def load_fixtures_with_rounds(path: str) -> pd.DataFrame:
    # Match Number,Round Number,Date,Location,Home Team,Away Team,Result
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={
        "Round Number": "Round",
        "Home Team": "HomeTeam",
        "Away Team": "AwayTeam",
        "Date": "DateTime",
    })
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce").astype("Int64")
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
    df["Date_key"] = df["DateTime"].dt.date.astype(str)  # YYYY-MM-DD
    df["HomeTeam"] = df["HomeTeam"].map(norm_team)
    df["AwayTeam"] = df["AwayTeam"].map(norm_team)
    return df[["Round", "Date_key", "HomeTeam", "AwayTeam"]].dropna()

def load_football_data_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    if "Date" not in df.columns or "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError("Football-data file must include Date, HomeTeam, AwayTeam columns.")

    df["Date_dt"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Date_key"] = df["Date_dt"].dt.date.astype(str)

    df["HomeTeam"] = df["HomeTeam"].map(norm_team)
    df["AwayTeam"] = df["AwayTeam"].map(norm_team)
    return df

def merge_rounds(fixtures_path: str, odds_path: str) -> pd.DataFrame:
    fx = load_fixtures_with_rounds(fixtures_path)
    od = load_football_data_odds(odds_path)

    merged = od.merge(
        fx,
        on=["Date_key", "HomeTeam", "AwayTeam"],
        how="left",
        validate="many_to_one",
    )

    merged = merged.dropna(subset=["Round"]).copy()
    merged["Round"] = merged["Round"].astype(int)
    return merged
