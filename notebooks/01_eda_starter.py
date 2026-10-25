# Paste into a Jupyter notebook (cells separated by # %%)
# %%
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_parquet("../data/processed/transactions_clean.parquet")
df.info(); df.describe(include="all").T.head(30)

# %% Missing values
(df.isna().mean().sort_values(ascending=False) * 100).head(15)

# %% Set these after inspecting your columns
PRICE, AREA = "price_aed", "area"   # rename to match your data

# %% Price distribution (log scale: property prices are heavily skewed)
sns.histplot(df[PRICE].dropna(), log_scale=True); plt.show()

# %% Median price by area (top 15 most active)
top = df[AREA].value_counts().head(15).index
df[df[AREA].isin(top)].groupby(AREA)[PRICE].median().sort_values().plot.barh(); plt.show()
