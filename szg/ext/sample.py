import numpy as np
from pymatgen.symmetry.bandstructure import HighSymmKpath

from szg.ext.grid import mesh3D


def sample_around_symmetry_points(structure):
    kpath = HighSymmKpath(structure)
    labels = kpath.kpath['path'][0]
    h_kpoints = [kpath.kpath['kpoints'][name] for name in labels]

    samples = mesh3D(0.1, 10)
    samples = np.stack(samples).reshape(3, 10 * 10 * 10)
    samples = samples.transpose()

    kpoints = []

    for k in h_kpoints:
        kpoints.append(samples + k)

    kpoints = np.concatenate(kpoints)
    return kpoints
