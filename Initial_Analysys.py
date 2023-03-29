import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimpy import skim
import seaborn as sns

df = pd.read_csv('water_potability.csv')

df_1 = df[df['ph'].isna()]
df_2 = df_1[df_1['Potability']==1]

df_2.shape

df_3 = df[df['Organic_carbon']>20]
df_3.shape

df_4 = df[df['ph']<6.5]

df_5 = df[df['Organic_carbon']>15]
df_5.shape
df_5.Potability.value_counts()

df.ph.isna().sum()

df.ph.hist()
plt.show()

# normal distribution - Will replace missing values with NA's
ph_mean = df.ph.mean()
df['ph'].fillna(value=ph_mean,inplace=True)

df.Sulfate.hist()
plt.show()

sul_mean = df.Sulfate.median()
df['Sulfate'].fillna(value=sul_mean,inplace=True)

skim(df)

df.Trihalomethanes.hist()
plt.show()

Tri_mean = df.Trihalomethanes.mean()
df.Trihalomethanes.fillna(value=Tri_mean,inplace=True)

skim(df)

df.to_csv('Cleaned_data.csv')

sns.boxplot(df.ph)
plt.show()

# Outlier treatment - Since this is a synthetic data, no real significance and also because the features are uncorrelated, we will clip the outlier values to better fit the model.

# Target variable distribution - Precision or recall? for accuracy.

# Standardized scaling for the features.

#