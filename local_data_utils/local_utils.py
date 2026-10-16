import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, os, umap
from PIL import Image
from typing import Union
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.style'] = 'italic'

__all__ = [
    'fig_to_image',
    'get_dir',
    'create_folder',
    'plot_2d',
    'plot_projection_grid_seaborn',
]

def fig_to_image(fig, dpi=100):

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)

    # --- SAFETY FIX ---
    # Ensure all metadata values are strings
    if hasattr(img, "info") and isinstance(img.info, dict):
        img.info = {k: str(v) for k, v in img.info.items()}

    return img

def get_dir(path):
    return '/'.join(path.split('/')[:-1])

def create_folder(folder_name: Union[str, os.PathLike]) -> str:

    folder_path = os.fspath(folder_name)
    if folder_path:
        if not os.path.isdir(folder_path):
            print(f'Creating Folder: {folder_path}')
            os.makedirs(folder_path, exist_ok=True)

    return folder_path

def plot_2d(matrices=None,
            colours=True,
            title='2D Scatter Plot',
            x_label='X',
            y_label='Y',
            plot_scatter=True, # 'continuous'
            fig_size=(8, 8),
            names=None,
            X=None,
            Y=None,
            scatter_labels = False,
            title_fontsize=25,
            linewidth=2,
            markers = ['o', '^', 'd', 's', 'p', 'X', 'v', '<', '>', 'h', '+', '*', 'x'],
            xticks = False,
            yticks = False,
            xtickslabels = False,
            ytickslabels = False,
            x_lim = None,
            y_lim = None,
            legend_loc = None,#'upper center',
            plot_legends = True,
            savepath='',
            return_img=False,
            legend_ncols=1,
            ):
    """
    Plot 1D or 2D matrices as 2D scatter plots.

    Args:
        matrices (list of np.ndarray): List of 1D or 2D arrays.
        colours (list of str or bool): List of color strings or True for default, False for none.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        fig_size (tuple): Size of the figure.
        names (list of str or None): Labels for annotations.
        X (np.ndarray): X-coordinates for annotations (if different from matrices).
        Y (np.ndarray): Y-coordinates for annotations (if different from matrices).
    """
    if matrices is None:
        raise ValueError("The 'matrices' argument must be provided and non-empty.")
    if not isinstance(matrices, list):
        matrices = [matrices]
        
    markers = markers*int(np.ceil(len(matrices)/len(markers)))
    markers = iter(markers)

    # Prepare color palette
    base_colors = list(OrderedDict.fromkeys(
        ['red', 'blue', 'green', 'maroon', 'magenta', 'darkorchid', 'darkorange', 'darkmagenta', 'brown', 'pink', 'gray', 'yellow'] +
        list(mcolors.TABLEAU_COLORS) +
        list(mcolors.XKCD_COLORS) +
        list(mcolors.CSS4_COLORS)
    ))
    excluded = {'white', 'snow', 'ghostwhite', 'ivory'}
    base_colors = [c for c in base_colors if c not in excluded]

    # Color iterator logic
    if isinstance(colours, list):
        color_iter = iter(colours)
    elif colours is True:
        color_iter = iter(base_colors)
    else:
        color_iter = iter(['black'] * len(matrices))

    # Set up plot
    fig = plt.figure(figsize=fig_size)
    plt.grid(color='gray', linestyle='--', linewidth=0.8)
    plt.title(title, fontsize=title_fontsize, fontweight='bold', color='darkgreen')

    # Plot each matrix
    for idx, matrix in enumerate(matrices):
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            x_vals = matrix
            y_vals = np.zeros_like(matrix)
        elif matrix.ndim == 2:
            if matrix.shape[1] == 2:
                x_vals = matrix[:, 0]
                y_vals = matrix[:, 1]
            elif matrix.shape[1] == 1:
                x_vals = matrix[:, 0]
                y_vals = np.zeros_like(x_vals)
            else:
                print(f"Skipping matrix at index {idx}: has unsupported shape {matrix.shape}")
                continue
        else:
            print(f"Skipping matrix at index {idx}: not 1D or 2D")
            continue
        
        if plot_scatter:
            if scatter_labels:
                plt.scatter(x_vals, y_vals, color=next(color_iter), label=scatter_labels[idx], s = title_fontsize*5, marker=next(markers), alpha=1.0)
            else:
                plt.scatter(x_vals, y_vals, color=next(color_iter), label=f"Set {idx+1}", s = title_fontsize*5, marker=next(markers), alpha=1.0)
        else:
            if scatter_labels:
                plt.plot(x_vals, y_vals, color=next(color_iter), label=scatter_labels[idx], linestyle='-', linewidth=linewidth, marker=next(markers), markersize = 5*linewidth)
            else:
                plt.plot(x_vals, y_vals, color=next(color_iter), label=f"Set {idx+1}", linestyle='-', linewidth=linewidth, marker=next(markers), markersize = 5*linewidth)

    # xmin_, xmax_ = plt.xlim()
    # x_ = np.linspace(xmin_, xmax_, 200)
    # # ±0.5 offset lines
    # plt.plot(x_, x_ + 0.5, color="green", linestyle="--", linewidth=2, label="±0.5 interval")
    # plt.plot(x_, x_ - 0.5, color="green", linestyle="--", linewidth=2)

    # Annotations (if applicable)
    if names:
        if X is None or Y is None:
            # Try to infer from the first valid matrix
            for m in matrices:
                m = np.asarray(m)
                if m.ndim == 2 and m.shape[1] == 2:
                    X, Y = m[:, 0], m[:, 1]
                    break
                elif m.ndim == 1:
                    X = m
                    Y = np.zeros_like(m)
                    break
        if X is not None and Y is not None:
            for name, x, y in zip(names, X, Y):
                plt.annotate(name, (x, y), textcoords="offset points", xytext=(0, -10), ha='center')

    if isinstance(xticks, (np.ndarray, list, tuple)):
        xticks = list(xticks)
        if isinstance(xtickslabels, (np.ndarray, list, tuple)):
            plt.xticks(xticks, xtickslabels if len(xticks) == len(xtickslabels) else xticks)
        else:
            plt.xticks(xticks, xticks)
    if isinstance(yticks, (np.ndarray, list, tuple)):
        yticks = list(yticks)
        if isinstance(ytickslabels, (np.ndarray, list, tuple)):
            plt.yticks(yticks, ytickslabels if len(yticks) == len(ytickslabels) else yticks)
        else:
            plt.yticks(yticks, yticks)

    # Axes labels
    plt.xlabel(x_label, labelpad=10, fontsize=title_fontsize-3)
    plt.ylabel(y_label, labelpad=10, fontsize=title_fontsize-3)
    plt.xticks(fontsize=title_fontsize-5)
    plt.yticks(fontsize=title_fontsize-5)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    
    if plot_legends:
        plt.legend(fontsize=title_fontsize-8, loc=legend_loc, frameon=True, ncol=legend_ncols, framealpha=0.7, edgecolor='white', facecolor='white')
    plt.tight_layout()
    
    if savepath:
        create_folder(get_dir(savepath))
        plt.savefig(savepath,  dpi=300)
        plt.close()
        print(f'Image saved in: {savepath}')
    elif return_img:
        return fig_to_image(fig, dpi=300)
    else:
        plt.show()


def plot_projection_grid_seaborn(
    matrices,
    labels,
    rep_names,
    metric='euclidean',
    cmap="viridis",
    highlight_index=None,
    normalise=True,
    show_xticks=False,
    show_yticks=False,
    show_grid=False,
    figsize=None,
    fontsize=15,
    savepath='',
    cbar_label='Fraction unbound value',
    suptitle = "2D Projections of Different Representations",
    proj_names=["PCA", "t-SNE", "UMAP", "Isomap",
                "FactorAnalysis", "MDS",
                "TruncatedSVD", "KernelPCA"],
):
    """
    Visualize multiple feature representations using a grid of 2D dimensionality
    reduction projections.

    Each row corresponds to one feature representation, while each column
    corresponds to a dimensionality reduction technique. Every feature matrix is
    standardized independently before projection. Each representation may contain
    a different number of samples.

    Parameters
    ----------
    matrices : list of ndarray
        List of feature matrices of shape (n_samples_i, n_features_i).

    labels : list of array-like
        List of label vectors, one for each representation. Each label vector
        must have the same length as the corresponding feature matrix.

    rep_names : list of str
        Names of the feature representations.

    highlight_index : int, optional
        Sample index to highlight. A point is highlighted only if the index
        exists for that representation.

    proj_names : list of str or "all", optional
        Projection methods to visualize.

    Returns
    -------
    None
    """

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

    matrices = [np.asarray(X) for X in matrices]
    labels = [np.asarray(y) for y in labels]

    if not (len(matrices) == len(labels) == len(rep_names)):
        raise ValueError("matrices, labels and rep_names must have the same length.")

    for X, y in zip(matrices, labels):
        if len(X) != len(y):
            raise ValueError("Each label vector must have the same number of samples as its corresponding feature matrix.")

    n_reps = len(matrices)
    if proj_names == "all":
        proj_names = ["PCA", "t-SNE", "UMAP", "Isomap", "FactorAnalysis", "MDS", "TruncatedSVD", "KernelPCA"]

    n_col = len(proj_names)
    fig, axes = plt.subplots(n_reps, 
                             n_col,
                             figsize=(4 * n_col, 3.3 * n_reps) if figsize is None else figsize,
                             squeeze=False,
                             )
    continuous_rows = []

    for i, (X, y, rep_name) in enumerate(zip(matrices, labels, rep_names)):

        Xs = StandardScaler().fit_transform(X) if normalise else X
        reducers = {
            "PCA": PCA(n_components=2, random_state=42),
            "t-SNE": TSNE(n_components=2, perplexity=10, learning_rate="auto", init="pca", random_state=42, metric=metric),
            "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, metric=metric),
            "Isomap": Isomap(n_components=2, n_neighbors=15),
            "FactorAnalysis": FactorAnalysis(n_components=2, random_state=42,),
            "MDS": MDS(n_components=2, random_state=42, n_init=1, max_iter=300,),
            "TruncatedSVD": TruncatedSVD(n_components=2, random_state=42,),
            "KernelPCA": KernelPCA(n_components=2, kernel="rbf", random_state=42,),
            }

        is_continuous = np.issubdtype(y.dtype, np.floating)

        if is_continuous:
            continuous_rows.append(y)

        for j, name in enumerate(proj_names):

            Z = reducers[name].fit_transform(Xs)
            ax = axes[i, j]

            if is_continuous:
                ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=35, alpha=0.85)
            else:
                sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette="tab10", ax=ax, s=35, alpha=0.85, linewidth=0, legend=False,)

            if (highlight_index is not None and highlight_index < len(Z)):
                ax.scatter(Z[highlight_index, 0], Z[highlight_index, 1], color="red", s=20, zorder=5)

            if i == 0:
                ax.set_title(name, fontsize=fontsize, fontweight="bold")

            if j == 0:
                ax.set_ylabel(rep_name, fontsize=fontsize, fontweight="bold",)
            else:
                ax.set_ylabel("")

            # ax.set_xlabel("")
            # ax.set_xticks([])
            # ax.set_yticks([])

            ax.set_xlabel("")
            if not show_xticks:
                ax.set_xticks([])

            if not show_yticks:
                ax.set_yticks([])
            ax.grid(show_grid)

    if len(continuous_rows) > 0:

        all_values = np.concatenate(continuous_rows)
        norm = plt.Normalize(all_values.min(), all_values.max(),)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([1.01, 0.12, 0.018, 0.76])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, fontsize=fontsize, fontweight="bold",)
        cbar.ax.tick_params(labelsize=fontsize)

    plt.suptitle(suptitle, fontsize=fontsize+5, fontweight="bold", y=1.02)
    plt.tight_layout()
    if savepath:
        create_folder(get_dir(savepath))
        plt.savefig(savepath,  dpi=300)
        plt.close()
        print(f'Image saved in: {savepath}')
    else:
        plt.show()