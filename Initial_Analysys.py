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



# Outlier treatment - Since this is a synthetic data, no real significance and also because the features are uncorrelated, we will clip the outlier values to better fit the model.
# https://datascience.stackexchange.com/questions/65802/for-outliers-treatment-clipping-winsorizing-or-removing
# Performing a 90% windorization window

sns.boxplot(df.ph)
plt.show()

Q1 = df['ph'].quantile(0.05)
Q2 = df['ph'].quantile(0.95)
df['ph'] = df['ph'].clip(lower = Q1, upper = Q2)

# Checking for outliers again
sns.boxplot(df.ph)
plt.show()

# Checking for outliers in Hardness
sns.boxplot(df.Hardness)
plt.show()

Q1 = df['Hardness'].quantile(0.05)
Q2 = df['Hardness'].quantile(0.95)
df['Hardness'] = df['Hardness'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Hardness)
plt.show()

# Checking for outliers in Solids
sns.boxplot(df.Solids)
plt.show()

Q1 = df['Solids'].quantile(0.05)
Q2 = df['Solids'].quantile(0.95)
df['Solids'] = df['Solids'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Solids)
plt.show()

# Checking for outliers in Chloramines
sns.boxplot(df.Chloramines)
plt.show()

Q1 = df['Chloramines'].quantile(0.05)
Q2 = df['Chloramines'].quantile(0.95)
df['Chloramines'] = df['Chloramines'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Chloramines)
plt.show()

# Checking for outliers in Sulfate
sns.boxplot(df.Sulfate)
plt.show()

Q1 = df['Sulfate'].quantile(0.05)
Q2 = df['Sulfate'].quantile(0.95)
df['Sulfate'] = df['Sulfate'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Sulfate)
plt.show()

# Checking for outliers in Conductivity
sns.boxplot(df.Conductivity)
plt.show()

Q1 = df['Conductivity'].quantile(0.05)
Q2 = df['Conductivity'].quantile(0.95)
df['Conductivity'] = df['Conductivity'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Conductivity)
plt.show()

# Checking for outliers in Organic_carbon
sns.boxplot(df.Organic_carbon)
plt.show()

Q1 = df['Organic_carbon'].quantile(0.05)
Q2 = df['Organic_carbon'].quantile(0.95)
df['Organic_carbon'] = df['Organic_carbon'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Organic_carbon)
plt.show()


# Checking for outliers in Trihalomethanes
sns.boxplot(df.Trihalomethanes)
plt.show()

Q1 = df['Trihalomethanes'].quantile(0.05)
Q2 = df['Trihalomethanes'].quantile(0.95)
df['Trihalomethanes'] = df['Trihalomethanes'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Trihalomethanes)
plt.show()

# Checking for outliers in Turbidity
sns.boxplot(df.Turbidity)
plt.show()

Q1 = df['Turbidity'].quantile(0.05)
Q2 = df['Turbidity'].quantile(0.95)
df['Turbidity'] = df['Turbidity'].clip(lower = Q1, upper = Q2)

# Checking again
sns.boxplot(df.Turbidity)
plt.show()

df.to_csv('Outlier_treated_data')

# Target variable distribution - Precision or recall? for accuracy.

# Standardized scaling for the features.
