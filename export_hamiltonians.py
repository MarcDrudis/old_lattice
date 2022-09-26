# 0. Standard imports
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
# from qiskit.aqua import Operator
import qutip as qt

from lattice import SquareLattice
from lattice.operators.qiskit_aqua_operator_utils import *

from lattice.wilson_fermions import dirac, build_hamilton, terms_hamilton_hopp_mass_wilson,hamilton_plaquette,hamilton_gauge,gauss_law_regularizer

from lattice.wilson_fermions.variational_form import WilsonLGT, hopping_term
from lattice.wilson_fermions.states import *

lattice_simplest = SquareLattice([3], bc='closed')

print(sys.argv)
spin, m,t,r,a,e,lam = sys.argv[1:]




S = float(spin)
ms = 0.
rep = dirac
params = {
    'm': float(m),#.5,
    't': float(t),
    'r': float(r),
    'a': float(a),#0.5,
    'e': float(e),#np.sqrt(2),
    'lam': float(lam),
    'S': float(S) }

boundary_cond = {
    (0, 0): ms,
    (lattice_simplest.nsites-1, 0): ms
}

#-----------------------------------------------------------------
hamiltonian = build_hamilton(lattice=lattice_simplest, 
                                rep=rep,
                                params=params, 
                                lam = params['lam'],
                                boundary_cond=boundary_cond,
                                output='qiskit')
# hamiltonian.chop()

# hamilton_qt = build_hamilton(lattice=lattice_simplest, 
#                                 rep=rep, 
#                                 params=params, 
#                                 lam = params['lam'],
#                                 boundary_cond=boundary_cond,
#                                 output='qutip')

# mass_qt,hopp_qt = terms_hamilton_hopp_mass_wilson(lattice_simplest,rep,params,'qutip')
# link_qt = (fermion_id(lattice_simplest)@ hamilton_gauge(lattice_simplest, params=params)).to_qubit_operator(output='qutip')
# regularizer_qt = gauss_law_regularizer(lattice_simplest, params,lam=params['lam'],boundary_cond=boundary_cond,output='qutip')

mass,hopp = terms_hamilton_hopp_mass_wilson(lattice_simplest,rep,params,'qiskit')
link = (fermion_id(lattice_simplest)@ hamilton_gauge(lattice_simplest, params=params)).to_qubit_operator(output='qiskit')
regularizer = gauss_law_regularizer(lattice_simplest, params,lam=params['lam'],boundary_cond=boundary_cond,output='qiskit')

mass.to_file('/home/drudis/Documents/StoredOps/mass.txt')
hopp.to_file('/home/drudis/Documents/StoredOps/hopp.txt')
link.to_file('/home/drudis/Documents/StoredOps/link.txt')
regularizer.to_file('/home/drudis/Documents/StoredOps/regularizer.txt')

assert hamiltonian == mass+hopp+link+regularizer , "Hamlitonian Error"

print("Success at creating old Hamiltonian")