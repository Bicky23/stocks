import pickle
import pandas as pd
import numpy as np

# load Linear Regression model
with open("models/linear_model.pkl", "rb") as f:
	model = pickle.load(f)

# dataframe
X_train = pd.read_csv("data/processed/X_train.csv")

## take input parameters
actual = float(input("Enter actual value:"))
forecast = float(input("Enter forecast value:"))
rolling_std = float(input("Enter rolling standard deviation:"))
z_score = float(input("Enter z-score:"))
one = float(input("Enter 1:"))
date = pd.to_datetime(input("Enter time of the year (YYYY-MM-DD):")).year

## lag features
actual_mean = np.mean(X_train["actual"])
forecast_mean = np.mean(X_train["forecast"])
rolling_std_mean = np.mean(X_train["rolling_std"])
z_score_mean = np.mean(X_train["z-score"])
open_mean = np.mean(X_train["1"])
lag_close_mean = np.mean(X_train["7_1"])

## load scalers
with open("models/1.pkl", "rb") as f:
    scaler_1 = pickle.load(f)
    one = scaler_1.transform(np.array(one).reshape(-1,1))[0][0]

with open("models/actual.pkl", "rb") as f:
    scaler_actual = pickle.load(f)
    actual = scaler_actual.transform(np.array(actual).reshape(-1,1))[0][0]

with open("models/forecast.pkl", "rb") as f:
    scaler_forecast = pickle.load(f)
    forecast = scaler_forecast.transform(np.array(forecast).reshape(-1,1))[0][0]

with open("models/z-score.pkl", "rb") as f:
    scaler_z_score = pickle.load(f)
    z_score = scaler_z_score.transform(np.array(z_score).reshape(-1,1))[0][0]

## two columns
diff = float(actual) - float(forecast)
year = pd.to_datetime(date).year
if year%2:
    year_even = 1
else:
    year_even = 0


series = pd.DataFrame({"actual":actual, "actual_1":actual_mean, "forecast":forecast, "forecast_1":forecast, "z-score":z_score, "z-score_1":z_score_mean,
			"1":one, "1_1":open_mean, "7_1":lag_close_mean, "rolling_std":rolling_std, "rolling_std_1":rolling_std_mean, "year_even":year_even,
			"difference":diff}, index=[0])


prediction = model.predict(series)[0][0]
print(prediction)