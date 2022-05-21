import os
import time
import numpy as np
import sympy as sp
import torch.nn

from os.path import join as pjoin
from torch.utils.data import DataLoader
from sympy.vector import CoordSys3D

from szg.core.hamiltonian import phase_term
from szg.core.kpoints import mp_kpath
from szg.dataset.kdataset import kDataset
from szg.ext.visualize import plot_bands
from szg.utils.timeutil import time_str


def fit_progressively(structure, k_vec, Ek, s_idx, e_idx, fitter, loss_func, optimizer, nbands, device):

    run_dir = pjoin('run', time_str(contain_second=True))
    os.mkdir(run_dir)

    npoints = e_idx - s_idx + 1
    for idx in range(npoints):
        kset = np.concatenate([k_vec[s_idx:s_idx + idx + 1]])
        Eset = np.concatenate([Ek[s_idx:s_idx + idx + 1]])

        dataset = kDataset(structure, kset, Eset)
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        if idx == 0:
            train(train_loader, fitter, loss_func, optimizer, nbands=nbands, n_epochs=100000, device=device)
        else:
            train(train_loader, fitter, loss_func, optimizer, nbands=nbands, n_epochs=100, device=device)

    predict(structure, fitter, nbands, device, run_dir)
    torch.save(fitter, pjoin(run_dir, 'model.pth'))


def train(train_loader, model, loss_func, optimizer, nbands, n_epochs, device):
    for epoch in range(n_epochs):
        loss_list = []
        t0 = time.time()
        for phase, energy in train_loader:
            phase, energy = phase.to(device), energy.to(device)
            pred = model(phase)

            loss = loss_func(energy, pred[:, :nbands])
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch}, loss:{np.array(loss_list).mean(): .6f}, '
              f'time cost:{time.time()-t0: .4f}s')
        if np.array(loss_list).mean() < 1e-5:
            break


def predict(structure, fitter, nbands, line_density=30, device=None, run_dir=None):
    phase = phase_term(structure, cutoff_radius=2.)

    k_vec, k_dist, k_node, labels = mp_kpath(structure, density=line_density)
    nk = len(k_vec)
    k = CoordSys3D('C')

    Ek = np.ones([nbands, nk])
    phase = sp.lambdify([k.x, k.y, k.z], phase, 'numpy')

    for ik in range(nk):
        phase_k = phase(k_vec[ik][0], k_vec[ik][1], k_vec[ik][2])
        phase_k = torch.complex(torch.FloatTensor(phase_k.real), torch.FloatTensor(phase_k.imag))
        phase_k = phase_k.to(device)

        pred = fitter(torch.unsqueeze(phase_k, dim=0))
        Ek[:,ik] = pred[0].detach().cpu().numpy()[:nbands]

    plot_bands(k_dist, k_node, labels, Ek, save_path=run_dir)
    return Ek
