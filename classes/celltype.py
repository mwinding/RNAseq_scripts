import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

class Celltype:
    def __init__(self, name, skids, color=None):
        self.name = name
        self.skids = list(np.unique(skids))
        if(color!=None):
            self.color = color

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)

    def get_color(self):
        return(self.color)


class Celltype_Analyzer:
    def __init__(self, list_Celltypes, adj=[]):
        self.Celltypes = list_Celltypes
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = []
        self.known_types_names = []
        self.adj = adj
        self.skids = [x for sublist in [celltype.get_skids() for celltype in list_Celltypes] for x in sublist]

    def add_celltype(self, Celltype):
        self.Celltypes = self.Celltypes + Celltype
        self.num += 1
        self.generate_adj()

    def set_known_types(self, list_Celltypes, unknown=True):

        if(list_Celltypes=='default'): data, list_Celltypes = Celltype_Analyzer.default_celltypes()
        if(unknown==True):
            unknown_skids = np.setdiff1d(self.skids, np.unique([skid for celltype in list_Celltypes for skid in celltype.get_skids()]))
            unknown_type = [Celltype('unknown', unknown_skids, 'tab:gray')]
            list_Celltypes = list_Celltypes + unknown_type
            
        self.known_types = list_Celltypes
        self.known_types_names = [celltype.get_name() for celltype in list_Celltypes]

    def get_known_types(self):
        return(self.known_types)

    # determine membership similarity (intersection over union) between all pair-wise combinations of celltypes
    def compare_membership(self, sim_type):
        iou_matrix = np.zeros((len(self.Celltypes), len(self.Celltypes)))

        for i in range(len(self.Celltypes)):
            for j in range(len(self.Celltypes)):
                if(len(np.union1d(self.Celltypes[i].skids, self.Celltypes[j].skids)) > 0):
                    if(sim_type=='iou'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        union = len(np.union1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        calculation = intersection/union
                    
                    if(sim_type=='dice'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff1 = len(np.setdiff1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff2 = len(np.setdiff1d(self.Celltypes[j].get_skids(), self.Celltypes[i].get_skids()))
                        calculation = intersection*2/(intersection*2 + diff1 + diff2)
                    
                    if(sim_type=='cosine'):
                            unique_skids = list(np.unique(list(self.Celltypes[i].get_skids()) + list(self.Celltypes[j].get_skids())))
                            data = pd.DataFrame(np.zeros(shape=(2, len(unique_skids))), columns = unique_skids, index = [i,j])
                            
                            for k in range(len(data.columns)):
                                if(data.columns[k] in self.Celltypes[i].get_skids()):
                                    data.iloc[0,k] = 1
                                if(data.columns[k] in self.Celltypes[j].get_skids()):
                                    data.iloc[1,k] = 1

                            a = list(data.iloc[0, :])
                            b = list(data.iloc[1, :])

                            dot = np.dot(a, b)
                            norma = np.linalg.norm(a)
                            normb = np.linalg.norm(b)
                            calculation = dot / (norma * normb)

                    iou_matrix[i, j] = calculation

        iou_matrix = pd.DataFrame(iou_matrix, index = [f'{x.get_name()} ({len(x.get_skids())})' for x in self.Celltypes], 
                                            columns = [f'{x.get_name()}' for x in self.Celltypes])

        return(iou_matrix)

    # calculate fraction of neurons in each cell type that have previously known cell type annotations
    def memberships(self, by_celltype=True, raw_num=False): # raw_num=True outputs number of neurons in each category instead of fraction
        fraction_type = np.zeros((len(self.known_types), len(self.Celltypes)))
        for i, knowntype in enumerate(self.known_types):
            for j, celltype in enumerate(self.Celltypes):
                if(by_celltype): # fraction of new cell type in each known category
                    if(raw_num==False):
                        if(len(celltype.get_skids())==0):
                            fraction = 0
                        if(len(celltype.get_skids())>0):
                            fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(celltype.get_skids())
                    if(raw_num==True):
                        fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))
                    fraction_type[i, j] = fraction
                if(by_celltype==False): # fraction of each known category that is in new cell type
                    fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(knowntype.get_skids())
                    fraction_type[i, j] = fraction

        fraction_type = pd.DataFrame(fraction_type, index = self.known_types_names, 
                                    columns = [f'{celltype.get_name()} ({len(celltype.get_skids())})' for celltype in self.Celltypes])
        
        if(raw_num==True):
            fraction_type = fraction_type.astype(int)
        
        return(fraction_type)

    def plot_memberships(self, path, figsize, rotated_labels = True, raw_num = False, memberships=None, ylim=None):
        if(type(memberships)!=pd.DataFrame):
            memberships = self.memberships(raw_num=raw_num)
        celltype_colors = [x.get_color() for x in self.get_known_types()]

        # plot memberships
        ind = [cell_type.get_name() for cell_type in self.Celltypes]
        f, ax = plt.subplots(figsize=figsize)
        plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
        bottom = memberships.iloc[0, :]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i, :]

        if(rotated_labels):
            plt.xticks(rotation=45, ha='right')
        if(ylim!=None):
            plt.ylim(ylim[0], ylim[1])
        plt.savefig(path, format='pdf', bbox_inches='tight')

    
    def upset_members(self, threshold=0, path=None, plot_upset=False, show_counts_bool=True, exclude_singletons_from_threshold=False, threshold_dual_cats=None, exclude_skids=None):

        celltypes = self.Celltypes

        contents = {} # empty dictionary
        for celltype in celltypes:
            name = celltype.get_name()
            contents[name] = celltype.get_skids()

        data = from_contents(contents)

        # identify indices of set intersection between all data and exclude_skids
        if(exclude_skids!=None):
            ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
            inter = set(ind_dict).intersection(exclude_skids)
            indices = [ind_dict[x] for x in inter]
            data = data.iloc[np.setdiff1d(range(0, len(data)), indices)]

        unique_indices = np.unique(data.index)
        cat_types = [Celltype(' + '.join([data.index.names[i] for i, value in enumerate(index) if value==True]), 
                    list(data.loc[index].id)) for index in unique_indices]

        # apply threshold to all category types
        if(exclude_singletons_from_threshold==False):
            cat_bool = [len(x.get_skids())>=threshold for x in cat_types]
        
        # allows categories with no intersection ('singletons') to dodge the threshold
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats==None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | ('+' not in x.get_name()))) for x in cat_types]

        # allows categories with no intersection ('singletons') to dodge the threshold and additional threshold for dual combos
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats!=None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | ('+' not in x.get_name())) | (len(x.get_skids())>=threshold_dual_cats) & (x.get_name().count('+')<2)) for x in cat_types]

        cats_selected = list(np.array(cat_types)[cat_bool])
        skids_selected = [x for sublist in [cat.get_skids() for cat in cats_selected] for x in sublist]

        # identify indices of set intersection between all data and skids_selected
        ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
        inter = set(ind_dict).intersection(skids_selected)
        indices = [ind_dict[x] for x in inter]

        data = data.iloc[indices]

        # identify skids that weren't plotting in upset plot (based on plotting threshold)
        all_skids = [x for sublist in [cat.get_skids() for cat in cat_types] for x in sublist]
        skids_excluded = list(np.setdiff1d(all_skids, skids_selected))

        if(plot_upset):
            if(show_counts_bool):
                fg = plot(data, sort_categories_by = None, show_counts='%d')
            else: 
                fg = plot(data, sort_categories_by = None)

            if(threshold_dual_cats==None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}.pdf', bbox_inches='tight')
            if(threshold_dual_cats!=None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}_dual-threshold{threshold_dual_cats}.pdf', bbox_inches='tight')

        return (cat_types, cats_selected, skids_excluded)