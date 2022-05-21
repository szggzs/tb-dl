from os.path import join as pjoin
from matplotlib import pyplot as plt


def plot_bands(k_dist, k_node, label, E_k, title='band structure', save_path=None):

    fig, ax = plt.subplots(dpi=500)

    ax.set_xlim(k_node[0],k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)

    # add vertical lines at node positions
    for n in range(len(k_node)):
      ax.axvline(x=k_node[n], linewidth=.5, color='k')

    ax.set_title(title)
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")

    for iband in range(E_k.shape[0]):
        ax.plot(k_dist,E_k[iband], linewidth=.5)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(pjoin(save_path, f'{title}.png'))
    else:
        plt.show()
