from __future__ import annotations
import os, re
from flask import Flask, render_template, request
import pandas as pd
import json
from plotly.utils import PlotlyJSONEncoder

from config import GOALS_CSV, THROWS_ROOT, THROWS_PATTERN

TEAM_MAP = {"ISR": "Israel", "CAN": "Canada", "TUR": "Turkey", "CHI": "China", "BRA": "Brazil"}

app = Flask(__name__)

# -------------------- utils --------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def safe_read_throws(root_folder: str, pattern: str = "**/*_Throws_data.xlsx") -> pd.DataFrame:
    import glob
    search_path = os.path.join(root_folder, pattern)
    files = glob.glob(search_path, recursive=True)
    parts = []
    for f in files:
        try:
            x = pd.read_excel(f)
            x["__source_filpe"] = os.path.basename(f)
            if "Game" not in x.columns or x["Game"].isna().all():
                game_name = os.path.basename(f).replace("_Throws_data.xlsx", "")
                x["Game"] = game_name
            parts.append(x)
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace("\n", " ").replace("  ", " ") for c in df.columns]
    return df

def dropdown_options(values):
    try:
        return [{"label": v, "value": v} for v in sorted(pd.Series(values).dropna().unique().tolist())]
    except Exception:
        return []

# If ever needed elsewhere
def to_zone_num(v):
    if pd.isna(v): return None
    s = str(v).strip()
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"-?\d+", s)
        return int(m.group()) if m else None

# ----- Parse "Lower 6" / "Upper 10" from Excel text -----
BAND_RE = re.compile(r'(?i)\b(lower|upper)\b')
NUM_RE  = re.compile(r'(\d{1,2})')

def parse_band_zone(val):
    """
    Accept strings like 'Lower 6', 'upper-10', 'UPPER 3', 'Lower_9'
    Returns (band, zone) with band in {'Lower','Upper'} and zone in 1..10
    """
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    m_band = BAND_RE.search(s)
    band = m_band.group(1).title() if m_band else None
    m_num = NUM_RE.search(s)
    zone = int(m_num.group(1)) if m_num else None
    if zone is not None and not (1 <= zone <= 10):
        zone = None
    return band, zone

# -------------------- load dfs --------------------
goals_df  = norm_cols(safe_read_csv(GOALS_CSV))
throws_df = norm_cols(safe_read_throws(THROWS_ROOT, THROWS_PATTERN))

# ---------- goals: normalize + release time parsing ----------
rename_map = {}
for c in goals_df.columns:
    lc = c.lower().replace("_", " ").strip()
    if lc == "throw number": rename_map[c] = "Throw Number"
    if lc == "from zone":    rename_map[c] = "From Zone"
    if lc == "to zone":      rename_map[c] = "To Zone"
    if lc in ("throwing team","throw team","attacking team"): rename_map[c] = "Throwing Team"
    if lc in ("defending team","defense team"):               rename_map[c] = "Defending Team"
    if lc in ("release time","release_time"):                 rename_map[c] = "Release Time"
goals_df = goals_df.rename(columns=rename_map)

mmss_re   = re.compile(r"^\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*$")
hhmmss_re = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$")
def parse_release_time_to_seconds(v):
    if pd.isna(v): return None
    s = str(v).strip()
    m = hhmmss_re.match(s)
    if m:
        h = float(m.group(1)); m_ = float(m.group(2)); sec = float(m.group(3))
        frac = float("0." + m.group(4)) if m.group(4) else 0.0
        return h*3600 + m_*60 + sec + frac
    m = mmss_re.match(s)
    if m:
        m_ = float(m.group(1)); sec = float(m.group(2))
        frac = float("0." + m.group(3)) if m.group(3) else 0.0
        return m_*60 + sec + frac
    try:
        return float(s)
    except Exception:
        return None

if "Release Time" in goals_df.columns:
    goals_df["rt_seconds"] = goals_df["Release Time"].apply(parse_release_time_to_seconds)
else:
    goals_df["rt_seconds"] = None

# helpers
def extract_teams_from_game(game_str: str):
    if not isinstance(game_str, str) or "_-_" not in game_str:
        return None, None
    left, right_rest = game_str.split("_-_", 1)
    right = right_rest.split("_")[0]
    return left, right

if "Game" in goals_df.columns:
    goals_df["Home"], goals_df["Away"] = zip(*goals_df["Game"].apply(extract_teams_from_game))
    def pretty_game(g):
        if not isinstance(g, str) or "_-_" not in g: return str(g)
        l, rest = g.split("_-_", 1); r = rest.split("_")[0]
        return f"{TEAM_MAP.get(l,l)} vs {TEAM_MAP.get(r,r)} ({g})"
    goals_df["GameLabel"] = goals_df["Game"].apply(pretty_game)

# -------------------- routes --------------------
@app.route("/")
def index():
    games  = dropdown_options(goals_df["Game"]) if "Game" in goals_df.columns else []
    tteams = dropdown_options(goals_df["Throwing Team"]) if "Throwing Team" in goals_df.columns else []
    dteams = dropdown_options(goals_df["Defending Team"]) if "Defending Team" in goals_df.columns else []
    return render_template("index.html", games=games, tteams=tteams, dteams=dteams)

@app.route("/goals")
def goals_page():
    # multi-select (accept both new names and the old singles)
    sel_games  = request.args.getlist("games")  or request.args.getlist("game")
    sel_tteams = request.args.getlist("tteams") or request.args.getlist("tteam")
    sel_dteams = request.args.getlist("dteams") or request.args.getlist("dteam")

    df = goals_df.copy()
    if sel_games:
        df = df[df["Game"].isin(sel_games)]
    if sel_tteams:
        df = df[df["Throwing Team"].isin(sel_tteams)]
    if sel_dteams:
        df = df[df["Defending Team"].isin(sel_dteams)]

    # >>> Parse band/zone directly from the Excel text (e.g., "Lower 6")
    if "To Zone" in df.columns:
        df["band"], df["zone_num"] = zip(*df["To Zone"].apply(parse_band_zone))
    else:
        df["band"], df["zone_num"] = None, None

    # payloads for the template JS
    cols = ["Throwing Team", "To Zone", "zone_num", "Throw Number", "rt_seconds", "band"]
    for c in cols:
        if c not in df.columns: df[c] = None
    records = df[cols].dropna(subset=["Throw Number", "rt_seconds"], how="any").to_dict(orient="records")
    zone_records = df[["Throwing Team", "zone_num", "band"]].dropna(subset=["zone_num"]).to_dict(orient="records")

    kpis = {
        "total_records": int(len(df)),
        "unique_teams": int(df["Throwing Team"].nunique() if "Throwing Team" in df.columns else 0),
        "total_goals": int(len(df))
    }

    games  = dropdown_options(goals_df["Game"]) if "Game" in goals_df.columns else []
    tteams = dropdown_options(goals_df["Throwing Team"]) if "Throwing Team" in goals_df.columns else []
    dteams = dropdown_options(goals_df["Defending Team"]) if "Defending Team" in goals_df.columns else []
    header_img = "/static/images/goal.jpg"
    unique_teams = sorted(df["Throwing Team"].dropna().unique().tolist()) if "Throwing Team" in df.columns else []

    return render_template(
        "goals.html",
        kpis=kpis,
        games=games, tteams=tteams, dteams=dteams,
        sel_games=sel_games, sel_tteams=sel_tteams, sel_dteams=sel_dteams,
        header_img=header_img,
        goals_records=json.dumps(records, cls=PlotlyJSONEncoder),
        zone_records=json.dumps(zone_records, cls=PlotlyJSONEncoder),
        unique_teams=json.dumps(unique_teams, cls=PlotlyJSONEncoder),
    )

@app.route("/throws")
def throws_page():
    sel_games = request.args.getlist("games") or request.args.getlist("game")
    sel_teams = request.args.getlist("teams") or request.args.getlist("team")

    df = throws_df.copy()

    # unify team column name
    team_col = None
    for c in ["Throwing Team", "Team", "Attacking Team"]:
        if c in df.columns:
            team_col = c
            break
    if team_col is None:
        df["__team"] = "Unknown"
        team_col = "__team"

    # apply filters (multi)
    if sel_games and "Game" in df.columns:
        df = df[df["Game"].isin(sel_games)]
    if sel_teams:
        df = df[df[team_col].isin(sel_teams)]

    # >>> Parse band/zone from Excel for FROM and TO
    if "To Zone" in df.columns:
        df["band_to"], df["zone_to"] = zip(*df["To Zone"].apply(parse_band_zone))
    else:
        df["band_to"], df["zone_to"] = None, None

    if "From Zone" in df.columns:
        df["band_from"], df["zone_from"] = zip(*df["From Zone"].apply(parse_band_zone))
    else:
        df["band_from"], df["zone_from"] = None, None

    kpis = {"total_throws": int(len(df))}

    games = dropdown_options(throws_df["Game"]) if "Game" in throws_df.columns else []
    teams = dropdown_options(throws_df[team_col]) if team_col in throws_df.columns else []

    throw_records = df[[team_col]].rename(columns={team_col: "Throwing Team"}).to_dict(orient="records")
    zone_to_records = df[[team_col, "zone_to", "band_to"]].rename(
        columns={team_col: "Throwing Team", "zone_to": "zone", "band_to": "band"}
    ).dropna(subset=["zone"]).to_dict(orient="records")
    zone_from_records = df[[team_col, "zone_from", "band_from"]].rename(
        columns={team_col: "Throwing Team", "zone_from": "zone", "band_from": "band"}
    ).dropna(subset=["zone"]).to_dict(orient="records")

    header_img = "/static/images/throw.jpg"
    unique_teams = sorted(df[team_col].dropna().unique().tolist())

    return render_template(
        "throws.html",
        kpis=kpis,
        games=games, teams=teams,
        sel_games=sel_games, sel_teams=sel_teams,
        header_img=header_img,
        throw_records=json.dumps(throw_records, cls=PlotlyJSONEncoder),
        zone_to_records=json.dumps(zone_to_records, cls=PlotlyJSONEncoder),
        zone_from_records=json.dumps(zone_from_records, cls=PlotlyJSONEncoder),
        unique_teams=json.dumps(unique_teams, cls=PlotlyJSONEncoder),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
