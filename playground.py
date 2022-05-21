import numpy as np
import torch

import os
from os.path import join as pjoin

from pymatgen.core import Structure

from szg.utils.dl4tb import predict
from szg.ext.readband import read_bandstructure
from szg.metric.metric import mean_error

li = os.listdir('run')
onsites = []

structure = Structure.from_file('data/POSCAR_graphite')
device = torch.device('cpu')
nbands = 16

k_vec, k_dist, k_node, labels, Ek = read_bandstructure('data/POSCAR_graphite',
                                                       'data/POSCARmp-band.txt',
                                                       start=0,
                                                       nbands=nbands)

for d in li:
    if not os.path.exists(pjoin('run', d, 'model.pth')):
        continue
    fitter = torch.load(pjoin('run', d, 'model.pth'))
    onsites.append(fitter.onsite_energies.detach().numpy())

    E_pred = predict(structure, fitter, nbands, line_density=201, device=device)

    pred = E_pred[:, 350:451]
    gt = Ek[:, 350:451]

    err = mean_error(pred, gt)

    print(err)

onsites = np.stack(onsites)
