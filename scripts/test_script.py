# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import polyseq as pseq

# matplotlib inline
np.random.seed(42)

# %%
# only need to run once
pseq.io.download_example_data()

# %%
data = pseq.io.load_example()

data_brain = pseq.io.load_example("brain")
data_vnc = pseq.io.load_example("vnc")

data = pseq.concat(data_brain, data_vnc)

data

# %%
import os
genes_path = '/data/GSM4030613_genes_sample_30.tsv'
matrix_path = '/data/GSM4030613_matrix_sample_30.mtx.gz'
genes = pd.read_csv(os.getcwd() + genes_path, delimiter='\t', header=None)[1]
matrix = pseq.io.read_mtx(os.getcwd() + matrix_path, genes = genes)

#matrix.loc[matrix.loc[:, 'Fmr1']>0, 'Fmr1']
#matrix.loc[matrix.loc[:, 'Prosap']>0, 'Prosap']

sum(matrix.loc[:, 'Fmr1']>0)
sum(matrix.loc[:, 'Prosap']>0)

# %%
