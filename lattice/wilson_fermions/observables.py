#import sys 
#sys.path.append("../")
from .basic_operators import *
from .clifford import dirac, weyl, zache
from ..operators.qiskit_aqua_operator_utils import operator_sum
from ..lattice import Lattice

################################################################################################
# 1. Set up the charge operators
################################################################################################


def site_charge(site, lattice, e=1., S=0):
    """Constructs the observable corresponding to the charge at lattice site `site`."""
    summands = []

    # 1. Construct the individual summands of the charge operator by summing over the components at one site
    for alpha in range(2):
        summands.append(psidag(site, alpha, lattice) * psi(site, alpha, lattice))
    ##############aggiungo questo per capire perché prova 
    #site_charge(site=[0],lattice=lattice, S=0).to_qubit_operator(output="qutip"), vedi che dà la matrice giusta 
    # epr i vettori DiracState("b").construct_circuit("qutip")
    #summands.append((-1.)*fermion_id(lattice))
    qx = e * operator_sum(summands)

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
       
        qx = qx @ link_id(S, lattice)

    return qx

def site_charge2(site, lattice, e=1., S=0):
    """Constructs the observable corresponding to the charge at lattice site `site`."""
    summands = []
    
    # 1. Construct the individual summands of the charge operator by summing over the components at one site
    for alpha in range(2):
            summands.append(psidag(site, alpha, lattice) *  psi(site, alpha, lattice))
    
    summands.append((-1.)*fermion_id(lattice))
    qx = e * operator_sum(summands) 

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
       
        
        qx = (qx) @ link_id(S, lattice)

    return qx


def total_charge(lattice, e=1., S=0):
    """Constructs the observable corresponding to total charge on the lattice."""
    summands = []

    # 1. Construct the total charge as the sum of charges over all sites
    for site in lattice.sites:
        summands.append(site_charge(site, lattice, e))
    Qtot = operator_sum(summands)

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
        Qtot = Qtot @ link_id(S, lattice)

    return Qtot

################################################################################################
# 2. Set up the mass type operators
################################################################################################


def site_mass(site, lattice, rep=dirac, m=1., S=0):
    """Constructs the observable corresponding to the mass (in the low energy limit)
     at lattice site `site` . """
    gamma0 = rep['gamma0']

    summands = []
    # 1. sum over all spinor components at the given lattice site
    for alpha in range(2):
        for beta in range(2):
            summands.append(psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(site, beta, lattice))
    mx = m * operator_sum(summands)

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
        mx = mx @ link_id(S, lattice)

    return mx


def total_mass(lattice, rep=dirac, m=1., S=0):
    """Constructs the observable corresponding to the total mass (in the low energy limit) on the lattice."""

    # 1. sum the site_mass operators over all lattice sites
    summands = []
    for site in lattice.sites:
        summands.append(site_mass(site, lattice, rep, m))
    Mtot = operator_sum(summands)

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
        Mtot = Mtot @ link_id(S, lattice)

    return Mtot




################################################################################################
# 3. Set up the total momentum operator
################################################################################################

def total_momentum(lattice, component=0, a=1., S=0):
    """
    Constructs the observable corresponding to the total momentum of the fermionic
    field on the lattice.

    Args:
        lattice (Lattice):
        component (int): Component of the total momentum to be known
        a (float):
        S (float):

    Returns:

    """
    summands = []

    # 1. Sum over all lattice sites and fermionic components to get the total fermion momentum operator
    for x in lattice.sites:
        if lattice.is_boundary_along(x, component, direction='positive'):
            continue

        # 1.1 Get the next site in the direction of the momentum component which should be evaluated
        next_site = np.copy(x)
        next_site[component] += 1
        next_site = lattice.project(next_site)

        # 1.2 Sum over the spinor components
        for alpha in range(2):
            site_momentum = psidag(x, alpha, lattice) * psi(next_site, alpha, lattice) / (2. * a)
            summands.append(site_momentum)
            summands.append(site_momentum.dag())

    Ptot = operator_sum(summands)

    # 2. If we are working with interacting Wilson fermions (S>0), tensor with an identity on the spin register
    if S > 0:
        Ptot = Ptot @ link_id(S, lattice)

    return Ptot


# TODO: opperation **2 (pow) not supported by my operators yet
# def total_mass_squared(lattice, Hamiltonian, a=1.):
#     """
#     The total squared mass operator M_tot^2 = H^2 - P^2
#     """
#     return Hamiltonian ** 2 - total_momentum(lattice, a) ** 2
