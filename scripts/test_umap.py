#%%
import umap
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

embedding = umap.UMAP().fit_transform(digits.data)

#%%
# plot data

df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
sns.scatterplot(data = df, x='UMAP1', y='UMAP2', s=2)