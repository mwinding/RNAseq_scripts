# module to analyze single cell RNAseq data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import polyseq as pseq
import umap

class Seq_mat():

    def __init__(self, gene_path, seq_matrix_path, FB_names=False):
        if(FB_names==False):
            self.genes = pd.read_csv(gene_path, delimiter='\t', header=None)[1]
        if(FB_names==True):
            self.genes = pd.read_csv(gene_path, delimiter='\t', header=None)[0]
        self.matrix = pseq.io.read_mtx(seq_matrix_path, genes = self.genes)
        self.embedding = umap.UMAP().fit_transform(self.matrix.values)
        self.data = pd.DataFrame(self.embedding, columns=['UMAP1', 'UMAP2'])

    def find_expression(self, gene_list, threshold, neurons_only=True):
        
        gene_columns = []
        for gene in gene_list:
            gene_column = []
            for cell in self.matrix.index:
                if(neurons_only):
                    if((self.matrix.loc[cell, gene]>threshold) & (self.matrix.loc[cell, 'elav']>threshold)==True):
                        gene_column.append(gene)
                    else:
                        gene_column.append('other')

                if(neurons_only==False):
                    if(self.matrix.loc[cell, gene]>threshold):
                        gene_column.append(gene)
                    else:
                        gene_column.append('other')

            gene_columns.append(gene_column)

        for i, gene_column in enumerate(gene_columns):
            self.data[gene_list[i]] = gene_column
