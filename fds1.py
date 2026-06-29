import pandas as pd
df = pd.read_csv("C:/Users/Raihan/Downloads/titanic.csv")
print(df)
print(df.head()) #eda
print(df.tail()) #eda
print(df.info()) #eda
print(df['PassengerId']) 
print(df.loc[3, 'PassengerId']) 
print(df.loc[4]) 
print(df.describe()) #eda
print(df['PassengerId'].value_counts()) #eda
print(df.duplicated().sum()) #eda
print(df['PassengerId'].value_counts()) #eda
print(df.shape) #eda