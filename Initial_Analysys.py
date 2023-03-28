import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('water_potability.csv')

df_1 = df[df['ph'].isna()]
df_2 = df_1[df_1['Potability']==1]

df_2.shape