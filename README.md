# Grand_Challenge-
Group Project - Accenture 
 ### EDA AND CLEANING OF CARS DATA 
# Circolante_Lazio.cvs

 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. LOAD DATASET
# =========================
# Load the vehicle dataset
file_path = "Circolante_Lazio.csv"
df = pd.read_csv(file_path, low_memory=False)

print("First rows of the dataset")
print(df.head())

print("Dataset shape (rows, columns)")
print(df.shape)

print("Column names")
print(df.columns)

# =========================
# 2. STANDARDIZE COLUMN NAMES
# =========================
# Convert column names to lowercase and remove spaces
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("/", "_")
)

print("Cleaned column names:")
print(df.columns)

# =========================
# 3. DATA QUALITY CHECK
# =========================

# Check missing values
print("Missing values per column")
print(df.isna().sum())

# Check duplicate rows
print("Number of duplicate rows:")
print(df.duplicated().sum())

# Check unique values
print("Unique values per column")
print(df.nunique())

# =========================
# 4. CLEAN TEXT VARIABLES
# =========================
# Convert text columns to uppercase and remove extra spaces

text_cols = df.select_dtypes(include="object").columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()

# Replace invalid Excel values
df.replace("########", np.nan, inplace=True)

# =========================
# 5. CONVERT NUMERIC COLUMNS
# =========================
# Try converting possible numeric columns

numeric_candidates = df.columns

for col in numeric_candidates:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# =========================
# 6. HANDLE MISSING VALUES
# =========================

# Remove rows with missing critical variables
df_clean = df.dropna()

print("Dataset shape after removing missing values")
print(df_clean.shape)

# =========================
# 7. DESCRIPTIVE STATISTICS
# =========================
# Generate statistical summary

print("Descriptive statistics")
print(df_clean.describe())

# =========================
# 8. FUEL TYPE ANALYSIS
# =========================
# Show distribution of fuel types

if "fuel" in df_clean.columns:

    plt.figure(figsize=(10,5))
    df_clean["fuel"].value_counts().plot(kind="bar")
    plt.title("Fuel Type Distribution")
    plt.xlabel("Fuel Type")
    plt.ylabel("Count")
    plt.show()

# =========================
# 9. TOP VEHICLE BRANDS
# =========================
# Show most common vehicle brands

if "brand" in df_clean.columns:

    plt.figure(figsize=(10,5))
    df_clean["brand"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Vehicle Brands")
    plt.xlabel("Brand")
    plt.ylabel("Count")
    plt.show()

# =========================
# 10. ENGINE SIZE DISTRIBUTION
# =========================
# Plot histogram for engine size

if "engine_size" in df_clean.columns:

    plt.figure(figsize=(10,5))
    plt.hist(df_clean["engine_size"], bins=30)
    plt.title("Engine Size Distribution")
    plt.xlabel("Engine Size")
    plt.ylabel("Frequency")
    plt.show()

# =========================
# 11. ANOMALY DETECTION
# =========================
# Detect outliers using the IQR method

if "engine_size" in df_clean.columns:

    Q1 = df_clean["engine_size"].quantile(0.25)
    Q3 = df_clean["engine_size"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean["engine_size"] < lower) |
        (df_clean["engine_size"] > upper)
    ]

    print("Number of anomalies detected:")
    print(len(outliers))

# =========================
# 12. SAVE CLEANED DATASET
# =========================
# Export the cleaned dataset

df_clean.to_csv("Circolante_Lazio_cleaned.csv", index=False)

print("Cleaned dataset saved.")

# Distribuzione Parco Veicoli per Anno, Comune Capoluogo e Categoria. Categorie AB,AM,AS,AV,MC,MM,MS,ND,RM,RS,TS

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
file_path = "Distribuzione Parco Veicoli per Anno, Comune Capoluogo e Categoria. Categorie AB,AM,AS,AV,MC,MM,MS,ND,RM,RS,TS.csv"
df = pd.read_csv(file_path)

print("FIRST 5 ROWS")
print(df.head())
print("\nSHAPE")
print(df.shape)
print("\nCOLUMNS")
print(df.columns.tolist())
print("\nINFO")
print(df.info())

# =========================
# 2. DATA QUALITY CHECK
# =========================
print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDUPLICATES")
print("Number of duplicate rows:", df.duplicated().sum())

print("\nUNIQUE YEARS")
print(df["Anno"].unique())

print("\nUNIQUE CITIES SAMPLE")
print(df["Comune"].unique()[:20])

# =========================
# 3. CLEANING
# =========================
# Fill missing year values using forward fill
df["Anno"] = df["Anno"].fillna(method="ffill")

# Convert year to integer
df["Anno"] = df["Anno"].astype(int)

# Numeric columns
numeric_cols = ["AB", "AM", "AS", "AV", "MC", "MM", "MS", "ND", "RM", "RS", "TS", "Totale"]

# Convert values like 2.059 -> 2059
# The file uses dots as thousand separators, not decimals
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(".", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nAFTER CLEANING - MISSING VALUES")
print(df.isna().sum())

print("\nAFTER CLEANING - DATA TYPES")
print(df.dtypes)

# =========================
# 4. STRUCTURE THE DATA
# =========================
# Separate yearly total rows from city-level rows
total_rows = df[df["Comune"].str.upper() == "TOTALE"].copy()
city_rows = df[df["Comune"].str.upper() != "TOTALE"].copy()

print("\nCITY-LEVEL DATA SHAPE")
print(city_rows.shape)

print("\nYEARLY TOTAL ROWS SHAPE")
print(total_rows.shape)

# Check duplicate city-year combinations
dup_city_year = city_rows.duplicated(subset=["Anno", "Comune"]).sum()
print("\nDUPLICATE CITY-YEAR ROWS:", dup_city_year)

# =========================
# 5. DESCRIPTIVE STATISTICS
# =========================
print("\nDESCRIPTIVE STATISTICS - CITY LEVEL")
print(city_rows[numeric_cols].describe())

# Top cities by total vehicles
top_cities = city_rows.groupby("Comune")["Totale"].max().sort_values(ascending=False).head(10)
print("\nTOP 10 CITIES BY TOTAL VEHICLES")
print(top_cities)

# Yearly totals
yearly_totals = city_rows.groupby("Anno")["Totale"].sum().reset_index()
print("\nYEARLY TOTAL VEHICLES")
print(yearly_totals)

# =========================
# 6. ANOMALY CHECK
# =========================
# IQR method for Totale
Q1 = city_rows["Totale"].quantile(0.25)
Q3 = city_rows["Totale"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = city_rows[(city_rows["Totale"] < lower_bound) | (city_rows["Totale"] > upper_bound)]

print("\nANOMALIES / OUTLIERS IN 'Totale'")
print(outliers[["Anno", "Comune", "Totale"]].sort_values("Totale", ascending=False).head(20))

# =========================
# 7. EDA VISUALIZATIONS
# =========================

# 1) Line chart: yearly total vehicles
plt.figure(figsize=(10, 5))
plt.plot(yearly_totals["Anno"], yearly_totals["Totale"], marker="o")
plt.title("Total Number of Vehicles by Year")
plt.xlabel("Year")
plt.ylabel("Total Vehicles")
plt.grid(True)
plt.show()

# 2) Bar chart: top 10 cities
plt.figure(figsize=(12, 6))
top_cities.sort_values().plot(kind="barh")
plt.title("Top 10 Cities by Total Vehicles")
plt.xlabel("Total Vehicles")
plt.ylabel("City")
plt.show()

# 3) Histogram of total vehicles
plt.figure(figsize=(10, 5))
plt.hist(city_rows["Totale"], bins=30)
plt.title("Distribution of Total Vehicles Across Cities")
plt.xlabel("Total Vehicles")
plt.ylabel("Frequency")
plt.show()

# 4) Boxplot for outliers
plt.figure(figsize=(8, 5))
plt.boxplot(city_rows["Totale"])
plt.title("Boxplot of Total Vehicles")
plt.ylabel("Total Vehicles")
plt.show()

# 5) Correlation matrix
corr = city_rows[numeric_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================
# 8. SIMPLE SUMMARY
# =========================
print("\nSUMMARY")
print("1. The dataset contains city-level vehicle data by year and category.")
print("2. Missing values were found mainly in the year column and were filled using forward fill.")
print("3. Numeric columns were cleaned by removing dots used as thousand separators.")
print("4. The total number of vehicles increases over time.")
print("5. Large cities such as Rome and Milan appear as outliers, but they are realistic outliers.")
