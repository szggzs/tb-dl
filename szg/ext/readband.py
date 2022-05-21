import numpy as np
import pandas as pd
from pymatgen.core import Structure

from szg.core.kpoints import mp_kpath


def read_bandstructure(struct_filename, band_filename, start=0, nbands=16):

    f = pd.read_csv(band_filename, header=None, sep='\t')
    structure = Structure.from_file(struct_filename)

    nk, blk = 1408, 4
    energies = np.array(f[1])

    Ek = np.zeros([nbands, nk])

    for idx in range(nbands):
        offset = (idx+start)*(nk+blk)
        Ek[idx, :] = energies[offset:offset+nk]

    k_vec, k_dist, k_node, labels = mp_kpath(structure, density=201)

    return k_vec, k_dist, k_node, labels, Ek
