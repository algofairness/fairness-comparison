import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# GERMAN
df_gen = pd.DataFrame.from_csv("results/german-gen.csv", index_col = 3)
df_calders = pd.DataFrame.from_csv("results/german-calders.csv", index_col = 3)
df_feldman = pd.DataFrame.from_csv("results/german-feldman.csv", index_col = 3)
df_kamishima = pd.DataFrame.from_csv("results/german-kamishima.csv", index_col = 3)
df_zafar = pd.DataFrame.from_csv("results/german-zafar.csv", index_col = 3)
frames = [df_gen, df_calders, df_feldman, df_kamishima, df_zafar]
df = pd.concat(frames)
sns.lmplot(x='Acc', y='DI', data=df,
           fit_reg=False,
           hue='Algorithms',
           scatter_kws={"s": 100},
           legend_out=False)
sns.plt.suptitle('German Data', fontsize=16)
plt.show()

# RICCI
df_gen = pd.DataFrame.from_csv("results/ricci-gen.csv", index_col = 3)
df_calders = pd.DataFrame.from_csv("results/ricci-calders.csv", index_col = 3)
df_feldman = pd.DataFrame.from_csv("results/ricci-feldman.csv", index_col = 3)
df_kamishima = pd.DataFrame.from_csv("results/ricci-kamishima.csv", index_col = 3)
df_zafar = pd.DataFrame.from_csv("results/ricci-zafar.csv", index_col = 3)
frames = [df_gen, df_calders, df_feldman, df_kamishima, df_zafar]
df = pd.concat(frames)
sns.lmplot(x='Acc', y='DI', data=df,
           fit_reg=False,
           hue='Algorithms',
           scatter_kws={"s": 100},
           legend_out=False)
sns.plt.suptitle('Ricci Data', fontsize=16)
plt.show()

# RETAILER
df_gen = pd.DataFrame.from_csv("results/retailer-gen.csv", index_col = 3)
df_calders = pd.DataFrame.from_csv("results/retailer-calders.csv", index_col = 3)
df_feldman = pd.DataFrame.from_csv("results/retailer-feldman.csv", index_col = 3)
df_kamishima = pd.DataFrame.from_csv("results/retailer-kamishima.csv", index_col = 3)
df_zafar = pd.DataFrame.from_csv("results/retailer-zafar.csv", index_col = 3)
frames = [df_gen, df_calders, df_feldman, df_kamishima, df_zafar]
df = pd.concat(frames)
sns.lmplot(x='Acc', y='DI', data=df,
           fit_reg=False,
           hue='Algorithms',
           scatter_kws={"s": 100},
           legend_out=False)
sns.plt.suptitle('Retailer Data', fontsize=16)
plt.show()

# ADULT
df_gen = pd.DataFrame.from_csv("results/adult-gen.csv", index_col = 3)
df_calders = pd.DataFrame.from_csv("results/adult-calders.csv", index_col = 3)
df_feldman = pd.DataFrame.from_csv("results/adult-feldman.csv", index_col = 3)
df_kamishima = pd.DataFrame.from_csv("results/adult-kamishima.csv", index_col = 3)
df_zafar = pd.DataFrame.from_csv("results/adult-zafar.csv", index_col = 3)
frames = [df_gen, df_calders, df_feldman, df_kamishima, df_zafar]
df = pd.concat(frames)
sns.lmplot(x='Acc', y='DI', data=df,
           fit_reg=False,
           hue='Algorithms',
           scatter_kws={"s": 100},
           legend_out=False)
sns.plt.suptitle('Adult Data', fontsize=16)
plt.show()
