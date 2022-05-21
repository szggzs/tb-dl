import numpy as np
import sympy as sp
import torch
import torch.nn
from sympy.vector import CoordSys3D

from szg.core.hamiltonian import eigenvalues
from szg.core.hamiltonian import phase_term


def energies_for_graphene(structure, n_orbitals, kpoints):

    h_dim = n_orbitals.sum()

    onsite_energies = torch.zeros(h_dim)
    hopping_energies = -torch.ones(1)

    phase = phase_term(structure, cutoff_radius=2.)
    k = CoordSys3D('C')
    phase = sp.lambdify([k.x, k.y, k.z], phase, 'numpy')

    energies = []
    for k_vec in kpoints:
        phase_k = phase(k_vec[0], k_vec[1], k_vec[2])
        phase_k = torch.complex(torch.FloatTensor(phase_k.real), torch.FloatTensor(phase_k.imag))
        e = eigenvalues(h_dim, n_orbitals, onsite_energies, hopping_energies, phase_k.unsqueeze(0))
        energies.append(e.numpy().squeeze(0))

    return energies
