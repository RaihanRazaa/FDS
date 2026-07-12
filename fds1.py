import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
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
print("PassengerId Column:\n", df["PassengerId"])
print("\nPassengerId at Index 3:", df.loc[3, "PassengerId"])
print("\nFull Row at Index 4:\n", df.loc[4])
print("\n--- Passenger ID Value Counts ---")
print(df["PassengerId"].value_counts())
binary = []
ordinal = []
nominal = []
numeric = []
for col in df.columns:
    if df[col].nunique(dropna=True) <= 2:
        binary.append(col)
    elif df[col].dtype == "object":
        nominal.append(col)
    elif np.issubdtype(df[col].dtype, np.number):
        numeric.append(col)
    else:
        ordinal.append(col)
print("\n--- Attribute Classification ---")
print("Binary attributes:", binary)
print("Ordinal attributes:", ordinal)
print("Nominal attributes:", nominal)
print("Numeric attributes:", numeric)

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
sns.countplot(x="Survived", data=df, ax=axes[0], palette="Set2")
axes[0].set_title("Survival Count")
sns.countplot(x="Sex", data=df, ax=axes[1], palette="Pastel1")
axes[1].set_title("Gender Distribution")
sns.countplot(x="Pclass", data=df, ax=axes[2], palette="Set1")
axes[2].set_title("Passenger Class")
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Age"], bins=20, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Age Distribution")
sns.histplot(df["Fare"], bins=20, kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Fare Distribution")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
