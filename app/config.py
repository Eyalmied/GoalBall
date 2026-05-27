from pathlib import Path

_HERE = Path(__file__).resolve().parent        # .../app/

GOALS_CSV     = str(_HERE / "data" / "GoalPredictions_AllGames.csv")
THROWS_ROOT   = str(_HERE / "data" / "Paralkympics2024")
THROWS_PATTERN = r"**/*_Throws_data.xlsx"
