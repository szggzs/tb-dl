import torch

from torch import nn

from szg.core.hamiltonian import eigenvalues


class TBFitter(nn.Module):
    def __init__(self, n_orbitals):
        super().__init__()

        self.n_orbitals = n_orbitals
        self.num_sites = len(n_orbitals)
        self.h_dim = n_orbitals.sum()

        h_idx = 0
        for idx in range(len(n_orbitals)):
            for jdx in range(idx + 1, len(n_orbitals)):
                h_idx += n_orbitals[idx]*n_orbitals[jdx]

        self.onsite_energies = nn.Parameter(torch.rand(self.h_dim))
        self.hopping_energies = nn.Parameter(torch.rand(h_idx))

    def forward(self, phase):

        energies = eigenvalues(self.h_dim, self.n_orbitals, self.onsite_energies, self.hopping_energies, phase)
        return energies
