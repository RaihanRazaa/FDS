import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def analyze_titanic():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    print("--- Full DataFrame ---")
    print(df)
    print(df.head())
    print(df.tail())
    df.info()
    print(df.describe())
    print(df.shape)
    print(df.isnull().sum())
    print(df.duplicated().sum())

    binary, ordinal, nominal, numeric = [], [], [], []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 2:
            binary.append(col)
        elif df[col].dtype == "object":
            nominal.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            numeric.append(col)
        else:
            ordinal.append(col)

    print("Binary:", binary)
    print("Ordinal:", ordinal)
    print("Nominal:", nominal)
    print("Numeric:", numeric)

    if len(nominal) >= 2:
        print("\nNominal Dissimilarity")
        for i in range(len(nominal)):
            for j in range(i+1, len(nominal)):
                v = df[nominal[i]].notna() & df[nominal[j]].notna()
                s1 = df.loc[v, nominal[i]].astype(str)
                s2 = df.loc[v, nominal[j]].astype(str)
                d = np.nan if len(s1)==0 else (s1!=s2).sum()/len(s1)
                print(f"{nominal[i]} vs {nominal[j]} = {d:.4f}")

    if len(numeric) >= 2:
        print("\nEuclidean Similarity")
        for i in range(len(numeric)):
            for j in range(i+1, len(numeric)):
                v = df[numeric[i]].notna() & df[numeric[j]].notna()
                s1 = df.loc[v, numeric[i]].astype(float)
                s2 = df.loc[v, numeric[j]].astype(float)
                if len(s1)==0:
                    sim=np.nan
                else:
                    if s1.max()!=s1.min():
                        s1=(s1-s1.min())/(s1.max()-s1.min())
                    if s2.max()!=s2.min():
                        s2=(s2-s2.min())/(s2.max()-s2.min())
                    sim=1/(1+euclidean(s1,s2))
                print(f"{numeric[i]} vs {numeric[j]} = {sim:.4f}")

    sns.countplot(x="Survived",data=df); plt.show()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm"); plt.show()

def analyze_automobile():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",header=None,na_values="?")
    df.columns=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

    print(df)
    print(df.head())
    print(df.tail())
    df.info()
    print(df.describe())
    print(df.shape)
    print(df.isnull().sum())
    print(df.duplicated().sum())

    binary, ordinal, nominal, numeric = [], [], [], []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 2:
            binary.append(col)
        elif df[col].dtype == "object":
            nominal.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            numeric.append(col)
        else:
            ordinal.append(col)

    print("Binary:", binary)
    print("Ordinal:", ordinal)
    print("Nominal:", nominal)
    print("Numeric:", numeric)

    if len(nominal)>=2:
        print("\nNominal Dissimilarity")
        for i in range(len(nominal)):
            for j in range(i+1,len(nominal)):
                v=df[nominal[i]].notna() & df[nominal[j]].notna()
                s1=df.loc[v,nominal[i]].astype(str)
                s2=df.loc[v,nominal[j]].astype(str)
                d=np.nan if len(s1)==0 else (s1!=s2).sum()/len(s1)
                print(f"{nominal[i]} vs {nominal[j]} = {d:.4f}")

    if len(numeric)>=2:
        print("\nEuclidean Similarity")
        for i in range(len(numeric)):
            for j in range(i+1,len(numeric)):
                v=df[numeric[i]].notna() & df[numeric[j]].notna()
                s1=df.loc[v,numeric[i]].astype(float)
                s2=df.loc[v,numeric[j]].astype(float)
                if len(s1)==0:
                    sim=np.nan
                else:
                    if s1.max()!=s1.min():
                        s1=(s1-s1.min())/(s1.max()-s1.min())
                    if s2.max()!=s2.min():
                        s2=(s2-s2.min())/(s2.max()-s2.min())
                    sim=1/(1+euclidean(s1,s2))
                print(f"{numeric[i]} vs {numeric[j]} = {sim:.4f}")

    sns.countplot(x="body-style",data=df); plt.xticks(rotation=45); plt.show()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm"); plt.show()

def main():
    choice=input("Choose a dataset:\n1. Automobile\n2. Titanic\nEnter choice: ").strip()
    match choice:
        case "1":
            analyze_automobile()
        case "2":
            analyze_titanic()
        case _:
            print("Invalid choice.")

if __name__=="__main__":
    main()
