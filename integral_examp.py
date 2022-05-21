from szg.core.orbitals import symbolic_orbitals
from szg.core.integral import cal_hopping_integral


r_A = [0., 0., 0.]
r_B = [1.4, 0., 0.]

orbitals_A, orbitals_B = symbolic_orbitals(r_A), symbolic_orbitals(r_B)
h_s_s = cal_hopping_integral(r_A, r_B, orbitals_A['s'], orbitals_B['s'])
