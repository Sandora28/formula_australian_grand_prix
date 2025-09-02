import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable caching
fastf1.Cache.enable_cache('cache')

# -------------------------------
# 1. Load 2024 Australian GP Race
# -------------------------------
race_2024 = fastf1.get_session(2024, 3, "R")
race_2024.load()

laps_2024 = race_2024.laps[["Driver","LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# Load 2024 Qualifying (for training input)
quali_2024 = fastf1.get_session(2024, 3, "Q")
quali_2024.load()
laps_quali_2024 = quali_2024.laps.groupby("Driver").apply(lambda x: x.pick_fastest())
laps_quali_2024 = laps_quali_2024.reset_index(drop=True)
laps_quali_2024 = laps_quali_2024[["Driver", "LapTime"]].copy()
laps_quali_2024["QualifyingTime (s)"] = laps_quali_2024["LapTime"].dt.total_seconds()

# Merge quali & race 2024 (Driver-based)
train_data = laps_quali_2024.merge(laps_2024, on="Driver")
X = train_data[["QualifyingTime (s)"]]
y = train_data["LapTime (s)"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

print(f"\nTraining error (MAE): {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# -------------------------------
# 2. Load 2025 Qualifying
# -------------------------------
quali_2025 = fastf1.get_session(2025, 3, "Q")
quali_2025.load()
laps_quali_2025 = quali_2025.laps.groupby("Driver").apply(lambda x: x.pick_fastest())
laps_quali_2025 = laps_quali_2025.reset_index(drop=True)
laps_quali_2025 = laps_quali_2025[["Driver","LapTime"]].copy()
laps_quali_2025["QualifyingTime (s)"] = laps_quali_2025["LapTime"].dt.total_seconds()

# Predict race lap times
laps_quali_2025["PredictedRaceTime (s)"] = model.predict(laps_quali_2025[["QualifyingTime (s)"]])

# -------------------------------
# 3. Load 2025 Race
# -------------------------------
race_2025 = fastf1.get_session(2025, 3, "R")
race_2025.load()
laps_race_2025 = race_2025.laps.groupby("Driver").apply(lambda x: x.pick_fastest())
laps_race_2025 = laps_race_2025.reset_index(drop=True)
laps_race_2025 = laps_race_2025[["Driver","LapTime"]].copy()
laps_race_2025["RaceTime (s)"] = laps_race_2025["LapTime"].dt.total_seconds()

# -------------------------------
# 4. Compare predictions vs real fastest laps
# -------------------------------
comparison = laps_quali_2025.merge(laps_race_2025, on="Driver", how="inner")
comparison = comparison[["Driver","QualifyingTime (s)","PredictedRaceTime (s)","RaceTime (s)"]]
comparison["PredictionError (s)"] = comparison["PredictedRaceTime (s)"] - comparison["RaceTime (s)"]
comparison = comparison.sort_values("PredictedRaceTime (s)")

# Print shrinked results
print("\n--- 2025 Fastest Lap Predictions vs Real ---\n")
print(comparison)

print(f"\n2025 Prediction Error (MAE): {comparison['PredictionError (s)'].abs().mean():.2f} s")