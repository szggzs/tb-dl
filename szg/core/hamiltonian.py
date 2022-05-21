import torch
import sympy as sp

from sympy.vector import CoordSys3D


def eigenvalues(h_dim, n_orbitals, onsite_energies, hopping_energies, phase):
    b = phase.shape[0]
    H_k = torch.zeros([b, h_dim, h_dim]) * (0 + 0.j)
    H_k = H_k.to(phase.device)

    r_cursor, h_cursor = 0, 0
    num_sites = len(n_orbitals)

    for idx in range(num_sites - 1):
        sub_r = n_orbitals[idx]
        c_cursor = n_orbitals[:idx + 1].sum()
        for jdx in range(idx + 1, num_sites):
            sub_c = n_orbitals[jdx]

            sub_matrix = hopping_energies[h_cursor:h_cursor + sub_r * sub_c].reshape([sub_r, sub_c])
            phase_term = phase[:, idx, jdx]
            phase_term = phase_term.unsqueeze(-1).unsqueeze(-1).repeat((1,) + sub_matrix.shape)
            sub_matrix = sub_matrix.repeat(b, 1, 1)

            sub_matrix = sub_matrix * phase_term
            H_k[:, r_cursor:r_cursor + sub_r, c_cursor:c_cursor + sub_c] += sub_matrix
            h_cursor += sub_r * sub_c
            c_cursor += n_orbitals[jdx]
        r_cursor += n_orbitals[idx]

    H_k = torch.diag(onsite_energies) + H_k + H_k.transpose(-1, -2).conj()
    eigens = torch.linalg.eigh(H_k)[0]
    return eigens


def phase_term(structure, cutoff_radius=2.):
    neighbors = []
    frac_coords = []
    sites = structure.sites
    num_sites = len(sites)

    for idx, s in enumerate(sites):
        frac_coords.append(s.frac_coords)

    for idx, s in enumerate(sites):
        fracs_c = frac_coords[:]
        fracs_c.pop(idx)
        neighbors.append(structure.get_neighbors(s, cutoff_radius))

    phase = sp.zeros(num_sites, num_sites)

    k = CoordSys3D('C')

    for idx in range(num_sites):
        phase[idx, idx] = 1
        for jdx in range(num_sites):
            for neighbor in neighbors[idx]:
                if neighbor.is_periodic_image(sites[jdx]):
                    d_vec = neighbor.frac_coords - sites[idx].frac_coords
                    phase[idx, jdx] += sp.exp(2 * sp.pi * (k.x * d_vec[0] + k.y * d_vec[1] + k.z * d_vec[2]) * (0 + 1j))

    return phase
