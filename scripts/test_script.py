# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import polyseq as pseq

# matplotlib inline
np.random.seed(42)

# %%
# test data
pseq.io.download_example_data()

data = pseq.io.load_example()
data_brain = pseq.io.load_example("brain")
data_vnc = pseq.io.load_example("vnc")

data = pseq.concat(data_brain, data_vnc)

data

# %%
# load real data

import os
genes_path = '/data/Sprecher_dataset/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_genes.tsv'
matrix_path = '/data/Sprecher_dataset/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_matrix.mtx.gz'
genes = pd.read_csv(os.getcwd() + genes_path, delimiter='\t', header=None)[1]
matrix = pseq.io.read_mtx(os.getcwd() + matrix_path, genes = genes)

#matrix.loc[matrix.loc[:, 'Fmr1']>0, 'Fmr1']
#matrix.loc[matrix.loc[:, 'Prosap']>0, 'Prosap']

sum((matrix.loc[:, 'Fmr1']>1) & (matrix.loc[:, 'elav']>1))
sum((matrix.loc[:, 'Prosap']>1) & (matrix.loc[:, 'elav']>1))
sum((matrix.loc[:, 'Prosap']>1) & (matrix.loc[:, 'Fmr1']>1) & (matrix.loc[:, 'elav']>1))
sum(matrix.loc[:, 'elav']>1)

# %%
# plot with UMAP

import umap

embedding = umap.UMAP().fit_transform(matrix.values)

# %%
# plot data
import seaborn as sns

df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
sns.scatterplot(data = df, x='UMAP1', y='UMAP2', s=2)
# %%
