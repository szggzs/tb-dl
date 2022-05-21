import numpy as np
import sympy as sp
import torch
from sympy.vector import CoordSys3D
from torch.utils.data import Dataset

from szg.core.hamiltonian import phase_term


class kDataset(Dataset):
    def __init__(self, structure, kpoints, energies):
        super().__init__()

        self.kpoints = kpoints

        phase = phase_term(structure, cutoff_radius=2.)
        k = CoordSys3D('C')
        phase = sp.lambdify([k.x, k.y, k.z], phase, 'numpy')

        k_term = []
        for idx in range(self.kpoints.shape[0]):
            k_point = self.kpoints[idx]
            k_term.append(phase(k_point[0], k_point[1], k_point[2]))

        self.k_term = np.stack(k_term)
        self.energies = energies

    def __getitem__(self, idx):
        term = self.k_term[idx]
        return torch.complex(torch.FloatTensor(term.real), torch.FloatTensor(term.imag)), \
               torch.FloatTensor(self.energies[idx])

    def __len__(self):
        return self.kpoints.shape[0]
