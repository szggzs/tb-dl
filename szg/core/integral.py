import numpy as np
import sympy as sp

from sympy.vector import CoordSys3D, Del
from scipy.integrate import nquad
from scipy.constants import hbar, m_e, e


def decouple(delta, psi1, psi2):

    l = np.dot(psi1, delta) * np.dot(psi2, delta)
    m = np.dot(psi1-np.dot(psi1, delta)*delta, psi2-np.dot(psi2, delta)*delta)
    return l, m


def cal_hopping_integral(r_A, r_B, orbital_A, orbital_B, Z=1, box_len=3):
    nabla = Del()

    C = CoordSys3D('C')
    symbols = [C.x, C.y, C.z]

    x, y, z = C.x - r_A[0], C.y - r_A[1], C.z - r_A[2]

    H_orbitalB = nabla.dot(nabla.gradient(orbital_B), doit=True)
    V = Z*e*e / (x * x + y * y + z * z) ** .5

    s_box = [[min(r_A[0], r_B[0])-box_len, max(r_A[0], r_B[0])+box_len],
             [min(r_A[1], r_B[1])-box_len, max(r_A[1], r_B[1])+box_len],
             [min(r_A[2], r_B[2])-box_len, max(r_A[2], r_B[2])+box_len]]
    A_H_B = nquad(sp.lambdify(symbols, orbital_A*H_orbitalB, 'numpy'), s_box, opts={'limit': 500})
    A_V_B = nquad(sp.lambdify(symbols, orbital_A*V*orbital_B, 'numpy'), s_box, opts={'limit': 500, 'points': (0, 0, 0)})

    h_ij = -hbar*hbar*A_H_B[0] / 2. / m_e + A_V_B[0]

    print(A_H_B, A_V_B)

    return h_ij
