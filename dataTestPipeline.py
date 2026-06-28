import pandas as pd
import numpy as np
import os
#from sklearn.preprocessing import OrdinalEncoder
#import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"input","dataTraining.csv")
file_path2 = os.path.join(script_dir,"input","stationDb.csv")
save_at = os.path.join(script_dir,"output","datasetForTesting.csv")

df = pd.read_csv(file_path,sep=",")

df["Actual departure date"] = pd.to_datetime(df["Actual departure date"])

df["Displayed flight number"] = df["Displayed flight number"].str.replace('[a-zA-Z]', '', regex=True)
df["departureStation"] = df["Actual leg ICAO"].str[:4]
df["arrivalStation"] = df["Actual leg ICAO"].str[-4:]

df2 = pd.read_csv(file_path2,sep=";")

df2["TRANSITION HOUR"] = df2["TRANSITION HOUR"].astype(str).str.replace(",", ".").astype(float)
df2["TRANSITION HOUR"] = df2["TRANSITION HOUR"].round(2)

fieldsToBeMerged = ["ICAO","TRANSITION HOUR"]
df2_unique = df2[fieldsToBeMerged].drop_duplicates(subset=["ICAO"], keep="first")
df = pd.merge(df, df2_unique, left_on="departureStation", right_on="ICAO",how="left")

df["TRANSITION HOUR"] = df["TRANSITION HOUR"].fillna(0)
df["LocalDateTime"] = df["Actual departure date"] + pd.to_timedelta(df["TRANSITION HOUR"], unit='h')
df["dateLocalTime"] = df["LocalDateTime"].dt.normalize()

df = df.rename(columns={
    "Flight great circle distance [SUM]": "flightGreatCircleDistance",
    "Aircraft maximum fuel flow [SUM]": "aircraftMaximumFuelFlow",
    "Aircraft minimum fuel flow [SUM]": "aircraftMinimumFuelFlow",
    "ZFWE Actual ZFW [SUM]": "actualZfw",
    "Block duration [SUM]": "blockDuration",
    "Fuel uplift in volume [SUM]": "fuelUpliftVolumeSkybreathe",
    "Aircraft registration": "aircraftRegistration"
})

collection = ["fuelUpliftVolumeSkybreathe","flightGreatCircleDistance",
              "aircraftMaximumFuelFlow","aircraftMinimumFuelFlow","actualZfw"]
for field in collection:
    df[field] = df[field].round(2)

conditions2 = [df["aircraftMinimumFuelFlow"] == 1600,
               df["aircraftMinimumFuelFlow"] == 1800,
               df["aircraftMinimumFuelFlow"] == 300]

choices2 = ["ceo320","neo320","atr"]

df["aircraftType"] = np.select(conditions2,choices2,default="0")

df["union"] = df["aircraftType"] + "." + df["Displayed flight number"] +  "." + df["Actual leg ICAO"]

df["blockDuration"] = pd.to_timedelta(df["blockDuration"],errors="coerce")
df["blockDuration"] = df["blockDuration"] / pd.Timedelta(hours=1)
df["blockDuration"] = df["blockDuration"].round(2)

df["dayOfMonth"] = df["dateLocalTime"].dt.day
df["monthNbr"] = df["dateLocalTime"].dt.month
df["dayOfWeek"] = df["dateLocalTime"].dt.dayofweek
df["yearNbr"] = df["dateLocalTime"].dt.year

df.info()

df.to_csv(save_at,sep=";",index=False)
