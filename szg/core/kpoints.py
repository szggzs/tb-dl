import numpy as np

from pymatgen.symmetry.bandstructure import HighSymmKpath


def mp_kpath(structure, density=30):
    kpath = HighSymmKpath(structure)
    labels = kpath.kpath['path'][0]
    path = [kpath.kpath['kpoints'][name] for name in labels]
    labels = [rf'${name}$' for name in labels]

    nk = density * (len(labels) - 1) + 1

    k_vec, k_dist, k_node = k_path(path, nk, lattice_vec=structure.lattice.matrix)

    return k_vec, k_dist, k_node, labels


def preprocess_mp_kpoints_sym(kpoints, k_labels):
    k_dist = np.zeros(len(kpoints))
    dist = 0
    k_node, labels = [], []

    for idx, k_vec in enumerate(kpoints):
        if idx > 0:
            dist += np.linalg.norm(k_vec - kpoints[idx - 1])
        k_dist[idx] = dist
        if k_labels[idx] is not '':
            if idx > 0 and k_labels[idx] == k_labels[idx - 1]:
                continue
            k_node.append(dist)
            labels.append(rf'${k_labels[idx]}$')

    return k_dist, k_node, labels


def preprocess_mp_kpoints(kpoints):
    k_dist = np.zeros(len(kpoints))
    dist = 0
    k_node, labels = [], []

    for idx, k_vec in enumerate(kpoints):
        if idx > 0:
            dist += np.linalg.norm(k_vec.cart_coords - kpoints[idx - 1].cart_coords)
        k_dist[idx] = dist
        if k_vec.label is not None:
            if idx > 0 and k_vec.label == kpoints[idx - 1].label:
                continue
            k_node.append(dist)
            labels.append(rf'${k_vec.label}$')

    return k_dist, k_node, labels


def k_path(kpts, nk, dim_k=3, lattice_vec=None, equals_in_node=True):

    # processing of special cases for kpts
    if kpts == 'full':
        # full Brillouin zone for 1D case
        k_list = np.array([[0.], [0.5], [1.]])
    elif kpts == 'fullc':
        # centered full Brillouin zone for 1D case
        k_list = np.array([[-0.5], [0.], [0.5]])
    elif kpts == 'half':
        # half Brillouin zone for 1D case
        k_list = np.array([[0.], [0.5]])
    else:
        k_list = np.array(kpts)

    # in 1D case if path is specified as a vector, convert it to an (n,1) array
    if len(k_list.shape) == 1 and dim_k == 1:
        k_list = np.array([k_list]).T

    # make sure that k-points in the path have correct dimension
    if k_list.shape[1] != dim_k:
        print('input k-space dimension is', k_list.shape[1])
        print('k-space dimension taken from model is', dim_k)
        raise Exception("\n\nk-space dimensions do not match")

    # must have more k-points in the path than number of nodes
    if nk < k_list.shape[0]:
        raise Exception("\n\nMust have more points in the path than number of nodes.")

    # number of nodes
    n_nodes = k_list.shape[0]

    # extract the lattice vectors from the TB model
    lat_per = np.copy(lattice_vec)
    # choose only those that correspond to periodic directions
    # lat_per = lat_per[self._per]
    # compute k_space metric tensor
    k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

    # Find distances between nodes and set k_node, which is
    # accumulated distance since the start of the path
    #  initialize array k_node
    k_node = np.zeros(n_nodes, dtype=float)
    for n in range(1, n_nodes):
        dk = k_list[n] - k_list[n - 1]
        dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
        k_node[n] = k_node[n - 1] + dklen

    # Find indices of nodes in interpolated list
    node_index = [0]
    if not equals_in_node:
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)
    else:
        for n in range(1, n_nodes - 1):
            node_index.append((nk - 1) // (n_nodes-1) * n)
        node_index.append(nk - 1)

    # initialize two arrays temporarily with zeros
    #   array giving accumulated k-distance to each k-point
    k_dist = np.zeros(nk, dtype=float)
    #   array listing the interpolated k-points
    k_vec = np.zeros((nk, dim_k), dtype=float)

    # go over all kpoints
    k_vec[0] = k_list[0]
    for n in range(1, n_nodes):
        n_i = node_index[n - 1]
        n_f = node_index[n]
        kd_i = k_node[n - 1]
        kd_f = k_node[n]
        k_i = k_list[n - 1]
        k_f = k_list[n]
        for j in range(n_i, n_f + 1):
            frac = float(j - n_i) / float(n_f - n_i)
            k_dist[j] = kd_i + frac * (kd_f - kd_i)
            k_vec[j] = k_i + frac * (k_f - k_i)

    return k_vec, k_dist, k_node
