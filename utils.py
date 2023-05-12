import os

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy


def get_adata(dataset, verbose=False):
    adata = scanpy.read_h5ad(os.path.join("data", f"{dataset}.h5ad"))
    if verbose:
        print(adata)
    return adata


def plot_cell_type_distribution(adata, save_to=None):
    sns.set_style('darkgrid')
    sns.set_theme('paper')
    df_counts = (adata.obs
                 .groupby(['condition', 'cell_type'])
                 .size()
                 .rename("count")
                 .to_frame()
                 .reset_index()
                 )
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,4), sharex=True, sharey=True)
    conditions = df_counts['condition'].drop_duplicates().to_list()
    for i, condition in enumerate(conditions):
        sns.barplot(data=df_counts[df_counts['condition'] == condition],
                    x='cell_type', y="count", ax=axes[i], alpha=0.5)
        axes[i].set_title(condition)
        axes[i].set_ylabel(None)
        axes[i].set_ylim([0, 3300])
        axes[i].set_xticklabels(
            axes[i].get_xticklabels(),
            rotation=90,
            horizontalalignment='right'
        )
        for k in axes[i].containers:
            axes[i].bar_label(k,)

    plt.tight_layout(h_pad=10)
    fig.subplots_adjust(top=0.88)
    fig.suptitle("Distribution of cells", fontsize=14)
    if save_to is None:
        save_to = os.path.join('figures', 'figure.png')

    plt.savefig(save_to, dpi=300)
    return fig
