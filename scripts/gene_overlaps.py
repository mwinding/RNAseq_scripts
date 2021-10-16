# %% 
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import classes.celltype as ct
import classes.RNAseq_tools as seq_tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import polyseq as pseq
import umap
import seaborn as sns

gene_path = 'data/Sprecher_dataset/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_genes.tsv'
seq_matrix_path = 'data/Sprecher_dataset/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_matrix.mtx.gz'

# will take ~1 minute
seq_data = seq_tools.Seq_mat(gene_path=gene_path, seq_matrix_path=seq_matrix_path, FB_names=True)

# only neurons
seq_data.matrix = seq_data.matrix[seq_data.matrix.loc[:, 'FBgn0260400']>0] #elav
seq_data.data = seq_data.data.loc[seq_data.matrix.index, :]


# %%
# setting up lists of genes of interest

threshold = 0

genelist = ['FBgn0266557', 'FBgn0031759', 'FBgn0026015', 'FBgn0028703', 'FBgn0033998', 'FBgn0053547', 'FBgn0031170', 'FBgn0061469', 'FBgn0005386', 'FBgn0038975', 
            'FBgn0040752', 'FBgn0083963', 'FBgn0031030', 'FBgn0011739', 'FBgn0013759', 'FBgn0013983', 'FBgn0261456', 'FBgn0263352', 'FBgn0266098', 'FBgn0083975',
            'FBgn0043362', 'FBgn0263289', 'FBgn0031866', 'FBgn0020251', 'FBgn0003256', 'FBgn0026317', 'FBgn0250786', 'FBgn0010333', 'FBgn0034136', 'FBgn0028734',
            'FBgn0000578', 'FBgn0040285'] 

genelist_names = ['kis', 'Kdm5', 'Top3B', 'Nhe3', 'row', 'Rim', 'Abca3', 'Ube3a', 'ash1', 'Nrx-1', 
                    'Prosap', 'Nlg3', 'Tao', 'wts', 'CASK', 'imd', 'hpo', 'Unr', 'rg', 'Nlg4',
                    'bchs', 'scrib', 'Nlg2', 'sfl', 'rl', 'Tsc1', 'Chd1', 'Rac1', 'DAT', 'Fmr1',
                    'ena', 'Scamp']

# upset plots
celltypes = [ct.Celltype(gene, seq_data.matrix[seq_data.matrix.loc[:, gene]>threshold].index) for gene in genelist]
celltypes = ct.Celltype_Analyzer(celltypes)
cat_types, _, _ = celltypes.upset_members(plot_upset=False, path='plots/genelist_upset.pdf')

# True/False annotations for each gene in the list
data = pd.concat([seq_data.data, (seq_data.matrix>threshold).loc[:, genelist]], axis=1)
data['overlap'] = data.iloc[:, 2:].sum(axis=1).values

# %%
# plot data
from pylab import figure, text

s=2
figsize = (4,4)
alpha = 0.8

# plot individual gene expressions separately
for i, gene in enumerate(genelist):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    sns.scatterplot(data=data, x='UMAP1', y='UMAP2', hue = gene, s=s, linewidth=0, alpha=alpha, legend=False, ax=ax)
    ax.text(0.5, 0.95, genelist_names[i],
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes, fontsize=12)
    fig.savefig(f'plots/{genelist_names[i]}_expression.pdf', format='pdf', bbox_inches='tight')

# summary plot of all gene expressions
n_rows = 6
n_cols = 6
fig, axs = plt.subplots(n_rows,n_cols, figsize = (2*len(genelist), 2*len(genelist)))
plt.subplots_adjust(wspace=0, hspace=0)
for i, gene in enumerate(genelist + ['','','','']):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[inds]
    if(gene==''):
        ax.axis('off')
    else:
        sns.scatterplot(data=data, x='UMAP1', y='UMAP2', hue = gene, s=16, linewidth=0, alpha=alpha, legend=False, ax=ax)
        ax.text(0.5, 0.9, genelist_names[i],
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=30)
        ax.axis('off')
fig.savefig('plots/all_expression_plot.pdf', format='pdf', bbox_inches='tight')

# overlap of gene expression plot
fig, ax = plt.subplots(1,1, figsize=figsize)
sns.scatterplot(data=data, x='UMAP1', y='UMAP2', hue='overlap', s=s, linewidth=0, alpha=0.5, ax=ax)
plt.legend(bbox_to_anchor=(1.025, 1), borderaxespad=0)
ax.set(xticks=[], yticks=[])
fig.savefig('plots/overlap_expression.pdf', format='pdf', bbox_inches='tight')

# overlap plot with threshold
data2 = data.copy()
data2.loc[data2.overlap<20, 'overlap']=0
fig, ax = plt.subplots(1,1, figsize=figsize)
sns.scatterplot(data=data2, x='UMAP1', y='UMAP2', hue = 'overlap', s=s, linewidth=0, alpha=alpha, ax=ax)
plt.legend(bbox_to_anchor=(1.025, 1), borderaxespad=0)
ax.set(xticks=[], yticks=[])
fig.savefig('plots/overlap_expression_20min-combo.pdf', format='pdf', bbox_inches='tight')

# %%
