import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv("dataset/disaster_relief_dataset.csv")

print("\n=============== Dataset Info ===============\n")
print(df.info())

print("\n=============== Missing Values ===============\n")
print(df.isnull().sum())

print("\n=============== Statistical Summary ===============\n")
print(df.describe())

print("\n=============== Disaster Type Counts ===============\n")
print(df["disaster_type"].value_counts())

#visualization of disaster type distribution(Count plot)
plt.figure(figsize=(10,5))
sns.countplot(data=df,x="disaster_type")

plt.title("Disaster Type Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#severity distribution visualization(Histogrm with KDE)
plt.figure(figsize=(8,5))
sns.histplot(data=df,x="disaster_severity",bins=10,kde=True)
plt.title("Disaster Severity Distribution")
plt.tight_layout()
plt.show()

#population vs food required visualization(Scatter Polt)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df,x="population_affected",y="food_required",hue="disaster_type")
plt.title("Population vs Food Required")
plt.tight_layout()
plt.show()

#Correlation(heatmap) 
numeric_df = df.select_dtypes(include = "number")
plt.figure(figsize = (14,10))
sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


