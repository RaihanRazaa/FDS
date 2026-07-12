import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", header=None, na_values='?')
df.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
              'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 
              'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
              'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

print("--- Full DataFrame ---")
print(df)
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Last 5 Rows ---")
print(df.tail())
print("\n--- DataFrame Information ---")
df.info()
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Shape (Rows, Columns) ---")
print(df.shape)
print("\n--- Missing Values Count ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows Count ---")
print(df.duplicated().sum())
print("\n--- Specific Data Selections ---")
if "price" in df.columns:
    print("Price Column:\n", df["price"])
    print("\nPrice at Index 3:", df.loc[3, "price"])
print("\nFull Row at Index 4:\n", df.loc[4])
if "price" in df.columns:
    print("\n--- Price Value Counts ---")
    print(df["price"].value_counts())

binary = []
ordinal = []
nominal = []
numeric = [] # Added numeric list
for col in df.columns:
    if df[col].nunique(dropna=True) <= 2:
        binary.append(col)
    elif df[col].dtype == "object":
        nominal.append(col)
    elif np.issubdtype(df[col].dtype, np.number): # Explicitly check for numeric
        numeric.append(col)
    else:
        ordinal.append(col)
print("\n--- Attribute Classification ---")
print("Binary attributes:", binary)
print("Ordinal attributes:", ordinal)
print("Nominal attributes:", nominal)
print("Numeric attributes:", numeric) # Print numeric attributes

# Nominal Attribute Dissimilarities
if len(nominal) >= 2:
    print("\n--- Nominal Attribute Dissimilarities ---")
    for i in range(len(nominal)):
        for j in range(i + 1, len(nominal)):
            col1 = nominal[i]
            col2 = nominal[j]
            s1 = df[col1].astype("object")
            s2 = df[col2].astype("object")
            valid = s1.notna() & s2.notna()
            s1 = s1[valid]
            s2 = s2[valid]
            if len(s1) == 0:
                dissim = np.nan
            else:
                mismatches = (s1 != s2).sum()
                dissim = mismatches / len(s1)
            print(f"{col1} vs {col2}: {dissim:.4f}")
    print("Note: simple matching dissimilarity = fraction of rows where values differ.")
else:
    print("\nNo valid nominal attribute pairs found for dissimilarity.")

# Numeric Attribute Dissimilarities
if len(numeric) >= 2:
    print("\n--- Numeric Attribute Dissimilarities ---")
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            col1 = numeric[i]
            col2 = numeric[j]
            s1 = df[col1]
            s2 = df[col2]
            valid = s1.notna() & s2.notna()
            s1 = s1[valid]
            s2 = s2[valid]
            if len(s1) == 0:
                dissim = np.nan
            else:
                diff = np.abs(s1 - s2)
                rng = max(s1.max() - s1.min(), s2.max() - s2.min())
                dissim = diff.mean() / rng if rng != 0 else diff.mean()
            print(f"{col1} vs {col2}: {dissim:.4f}")
    print("Note: normalized numeric dissimilarity = mean absolute difference / range.")
else:
    print("\nNo valid numeric attribute pairs found for dissimilarity.")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
if "body-style" in df.columns:
    sns.countplot(x="body-style", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Body Style Distribution")
else:
    axes[0].text(0.5, 0.5, "body-style column not found", ha='center', va='center')
if "fuel-type" in df.columns:
    sns.countplot(x="fuel-type", data=df, ax=axes[1], palette="Pastel1")
    axes[1].set_title("Fuel Type Distribution")
else:
    axes[1].text(0.5, 0.5, "fuel-type column not found", ha='center', va='center')
if "drive-wheels" in df.columns:
    sns.countplot(x="drive-wheels", data=df, ax=axes[2], palette="Set1")
    axes[2].set_title("Drive Wheels Distribution")
else:
    axes[2].text(0.5, 0.5, "drive-wheels column not found", ha='center', va='center')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if "horsepower" in df.columns:
    sns.histplot(df["horsepower"], bins=20, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Horsepower Distribution")
else:
    axes[0].text(0.5, 0.5, "horsepower column not found", ha='center', va='center')
if "price" in df.columns:
    sns.histplot(df["price"], bins=20, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Price Distribution")
else:
    axes[1].text(0.5, 0.5, "price column not found", ha='center', va='center')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
