from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo

day_create_model = datetime.now(ZoneInfo("Europe/Moscow")).date()

forest_reg = RandomForestRegressor(max_features=8, n_estimators=30, n_jobs=-1,
                      random_state=42)

df = pd.read_csv('final_data/housing_prepared.csv')

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"].copy()

forest_reg.fit(X, y)

joblib.dump(forest_reg, f"models/forest_reg_{day_create_model}.joblib")