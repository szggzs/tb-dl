import re

import sympy as sp
from sympy.vector import CoordSys3D


def get_valence_num(E):

    orbitals = E.electronic_structure
    orbitals = orbitals.split('.')

    if '[' in orbitals[0]:
        orbitals.pop(0)

    valence_number = 0
    for o in orbitals:
        number = re.findall(r'\d+', o)[-1]
        valence_number += int(number)

    return valence_number


def symbolic_orbitals(origin=None):
    if origin is None:
        origin = [0, 0, 0]
    C = CoordSys3D('C')
    x, y, z = C.x - origin[0], C.y - origin[1], C.z - origin[2]

    a = 1

    coeff_s = (2 * a / sp.pi) ** .75
    coeff_p = (128 * a ** 5 / sp.pi / sp.pi / sp.pi) ** .25

    r = (x * x + y * y + z * z) ** .5

    s = coeff_s * sp.exp(-a * r * r)
    px = coeff_p * x * sp.exp(-a * r * r)
    py = coeff_p * y * sp.exp(-a * r * r)
    pz = coeff_p * z * sp.exp(-a * r * r)

    orbits = {
        's': s,
        'px': px,
        'py': py,
        'pz': pz
    }

    return orbits
