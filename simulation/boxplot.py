import matplotlib as mpl
from matplotlib import gridspec
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pk
from matplotlib.patches import Patch
tasks = ['sim_dense_1.0','sim_dense_2.0','sim_sparse_1.0', 'sim_sparse_2.0']
tasksv = {'sim_sparse_1.0': r'Example 1 ($\sigma = 1$)', 'sim_sparse_2.0': r'Example 1 ($\sigma = 2$)',
          'sim_dense_1.0': r'Example 2 ($\sigma = 1$)','sim_dense_2.0': 'Example 2 ($\sigma = 2$)'}

strategys= ['raw', 'pair', 'pool', 'ace']
strategys_legend= ['Raw score', 'Paired score', 'Pooled score', 'ACE']
index = ['Davies-Bouldin', 'Calinski-Harabasz', 'Silhouette (cosine)', 'Silhouette (euclidean)']


strategy_colors = {
'raw': '#FFB6C1',
'pair': '#90EE90',
'pool':  'lightgrey',
'ace': '#00BFFF'}
colors = [strategy_colors['raw'], strategy_colors['pair'], strategy_colors['pool'], strategy_colors['ace']]
hatch_styles = ['///', '---', '||', '\\\\\\']
flierprops = dict(marker='o', color='black', alpha=0.5, markersize=1.5)  
medianprops = dict(color='darkred', linewidth=1.5)  
meanprops = {
    "marker": "^",  # shape 
    "markerfacecolor": "red",  # fill color 
    "markeredgecolor": "red",  # edge color 
    "markersize": 5
}
for ext in ['nmi', 'acc']:
    with open('eval/box_{}.pkl'.format(ext), 'rb') as op:
        data_load = pk.load(op)
    for task in tasks:
        data_t = data_load[task]
        for crit in ['tau', 'corr']:
            data_c = data_t[crit]
            fig, ax = plt.subplots(figsize=(12, 6))
            positions = []
            group_gap = 5  
            for i in range(4):
                positions.extend([i * group_gap + j for j in range(4)])
            for i, metric in enumerate(['dav', 'ch', 'cosine', 'euclidean']):
                data = data_c[metric]
                start = i * 4
                end = start + 4
                bp = ax.boxplot(data[:, ::-1], positions=positions[start:end], patch_artist=True,  boxprops=dict(linewidth=2, color="black"),
                                widths=0.75, flierprops=flierprops,  medianprops=medianprops, showmeans=True, meanprops=meanprops)  
                for patch, color, hatch in zip(bp['boxes'], colors, hatch_styles):
                    patch.set_facecolor(color)
                    patch.set_hatch(hatch)
                if i != 3:
                    ax.axvline(x= positions[end-1] + 1, color='gray', linestyle='--', linewidth=1)
            handles = [
                Patch(facecolor=color, edgecolor='black', hatch=hatch)
                for color, hatch in zip(colors, hatch_styles)
            ]
            ax.legend(handles, strategys_legend, loc = 'lower right', ncol = 1, fontsize = 12.5)
            plt.setp(ax.get_legend().get_texts(), fontweight='bold')  
            current_ylim = ax.get_ylim()
            new_ylim = (current_ylim[0] - 0.1, current_ylim[1])
            ax.set_ylim(new_ylim)
            ax.set_xticks([(group_gap * i) + 1.5 for i in range(4)])  
            ax.set_xticklabels(index, fontsize=14, fontweight='bold')
            ax.set_title(tasksv[task], fontsize=16, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2)  
                spine.set_color("black")  
            if crit == 'tau':
                if ext == 'acc':
                    ax.set_ylabel(r'Kendall rank correlation $\tau_{B}$ (vs. ACC)',fontsize = 12, fontweight='bold')
                else:
                    ax.set_ylabel(r'Kendall rank correlation $\tau_{B}$ (vs. NMI)',fontsize = 12, fontweight='bold')
            else:
                if ext == 'acc':
                    ax.set_ylabel(r"Spearman's rank correlation $r_s$ (vs. ACC)", fontsize=12, fontweight='bold')
                else:
                    ax.set_ylabel(r"Spearman's rank correlation $r_s$ (vs. NMI)",fontsize = 12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('{}_{}_{}.pdf'.format(ext, task, crit), transparent=True, bbox_inches='tight', pad_inches=0.05)
