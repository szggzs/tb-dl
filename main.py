import numpy as np
import torch.nn
import torch.optim as optim
from pymatgen.core import Structure

from szg.ext.readband import read_bandstructure
from szg.model.TBFitter import TBFitter
from szg.utils.dl4tb import fit_progressively

pos = Structure.from_file('data/POSCAR_graphite')
sites = pos.sites
n_orbitals = []
nbands = 16

for idx, s in enumerate(sites):
    valence_num = 5
    n_orbitals.append(valence_num)

n_orbitals = np.array(n_orbitals)

device = torch.device('cpu')
torch.cuda.set_device(0)

fitter = TBFitter(n_orbitals)
fitter.to(device)

optimizer = optim.Adam(fitter.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()

k_vec, k_dist, k_node, labels, Ek = read_bandstructure('data/POSCAR_graphite',
                                                       'data/POSCARmp-band.txt',
                                                       start=0,
                                                       nbands=nbands)
Ek = Ek.transpose(-1, -2)

for idx in range(50):
    fit_progressively(pos, k_vec, Ek, 350, 450, fitter, loss_func, optimizer, nbands, device)
