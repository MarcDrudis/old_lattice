import numpy as np
from ..operators.fermionic_operators import FermionicOperator, BaseFermionOperator
from .basic_sun_operators import *
from .string_hamiltonian import conserved_charge
from ..wilson_fermions.observables import *

################################################################################################
# 1. Set up the mass type operators
################################################################################################


def color_site_mass(site, lattice, ncolors=2, rep=dirac, m=1.):
    """Constructs the observable corresponding to the mass (in the low energy limit)
     at lattice site `site` . """
    gamma0 = rep['gamma0']

    summands = []
    # 1. sum over all spinor components at the given lattice site
    for color_idx in range(ncolors):
        for alpha in range(2):
            for beta in range(2):
                summands.append(psidag_color(site, color_idx, alpha, lattice, ncolors) * gamma0[alpha, beta] *
                                psi_color(site, color_idx, beta, lattice, ncolors))
    mx = m * operator_sum(summands)

    return mx


def color_total_mass(lattice, ncolors=2, rep=dirac, m=1.):
    """Constructs the observable corresponding to the total mass (in the low energy limit) on the lattice."""

    # 1. sum the site_mass operators over all lattice sites
    summands = []
    for site in lattice.sites:
        summands.append(color_site_mass(site, lattice, ncolors, rep, m))
    m_tot = operator_sum(summands)

    return m_tot

################################################################################################
# 2. Set up the "naive" charge operators
################################################################################################


def color_site_charge(site, lattice, color_idx, ncolors=2, g=1.):
    """Constructs the observable corresponding to the charge at lattice site `site`."""
    summands = []

    # 1. Construct the individual summands of the charge operator by summing over the components at one site
    for alpha in range(2):
        summands.append(psidag_color(site, color_idx, alpha, lattice, ncolors) *
                        psi_color(site, color_idx, alpha, lattice, ncolors))
    qx = g * operator_sum(summands)

    return qx


def color_total_charge(lattice, color_idx, ncolors=2, g=1.):
    """Constructs the observable corresponding to total charge on the lattice."""
    summands = []

    # 1. Construct the total charge as the sum of charges over all sites
    for site in lattice.sites:
        summands.append(color_site_charge(site, lattice, color_idx, ncolors, g))
    q_tot = operator_sum(summands)

    return q_tot

################################################################################################
# 3. Set up the conserved charge operators
################################################################################################


def gen_site_charge(site, lattice, generator, g=1.):
    """Constructs the observable Q_{site} = g * psidag_{site} * T * psi_{site}, where T is one generator
    of the gauge group, corresponding to the conservced charge at lattice site `site`."""

    # 1. Construct the individual summands of the charge operator by summing over the components at one site
    return g * conserved_charge(site, generator, lattice)


def gen_total_charge(lattice, generator, g=1.):
    """Constructs the observable Q_{tot} = g * sum_{sites} psidag_{site} * T * psi_{site}, where T is one generator
    of the gauge group, corresponding to total conserved charge on the lattice."""
    summands = []

    # 1. Construct the total charge as the sum of charges over all sites
    for site in lattice.sites:
        summands.append(gen_site_charge(site, lattice, generator, g))
    q_tot = operator_sum(summands)

    return q_tot
