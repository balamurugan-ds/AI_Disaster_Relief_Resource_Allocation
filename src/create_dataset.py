import pandas as pd
import numpy as np 
import os

np.random.seed(42)

n = 5000

disaster_types =[
    'Flood',
    'Earthquake',
    'Cyclone',
    'Landslide',
    'Drought',
    'wildfire',
]

locations = [
    "Chennai",
    "Coimbatore",
    "Thiruppur",
    "Madurai",
    "Trichy",
    "Salem",
    "Erode",
    "Thanjavur",
    "Tirunelveli",
    "Cuddalore",
]

df = pd.DataFrame({
    "disaster_type":np.random.choice(disaster_types, n),
    "location":np.random.choice(locations, n),
    "population_affected":np.random.randint(500,50000, n),
    "disaster_severity":np.random.randint(1,11,n),
    "injuried_people":np.random.randint(0,3000,n),
    "displaced_people":np.random.randint(100,30000,n),
    "duration_days":np.random.randint(1,31,n),
    "temperature":np.random.randint(0,200,n).round(1),
    "rainfall":np.random.randint(0,500,n).round(1),
    "infrastructure_damage":np.random.randint(0,100,n).round(1)
})

#Resource formula predicting and variable assignments
severity = df['disaster_severity']
population = df['population_affected']
displaced = df['displaced_people']
injuried = df['injuried_people']

#required resources calculation
df["food_required"] = (population * 1.5 + displaced * 1.2 + severity * 500 + np.random.normal(0,500,n)).clip(100).astype(int)
df["water_required"] = (population *3 + displaced * 2 + severity * 1000 +np.random.normal(0,1000,n)).clip(500).astype(int)
df["medical_supplies_required"] = (injuried * 2 + displaced * 100 + severity * 0.05 + np.random.normal(0,100,n)).clip(50).astype(int)
df["rescue_teams_required"] = (severity * 0.8 + population /5000 + injuried /500 + np.random.normal(0,1,n)).clip(1).astype(int)
df["shelter_required"] = (displaced * 1.05 + + severity * 100 + np.random.normal(0,200,n)).clip(100).astype(int)

#create a dataset 
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/disaster_relief_dataset.csv",index=False)

#output the first few rows of the dataset
print(df.head())
print("Dataset created and saved successfully")
print("\n shape of the dataset:",df.shape)