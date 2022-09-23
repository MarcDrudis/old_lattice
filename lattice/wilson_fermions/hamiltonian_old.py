from .basic_operators import *
from .clifford import dirac, weyl, zache
from ..operators.qiskit_aqua_operator_utils import operator_sum

################################################################################################
# 1. Set up the mass term
################################################################################################

def hamilton_mass(lattice, rep=dirac, m=1.):
    """
    This function contructs the mass term of the `Wilson` Hamiltonian for a two component
    dirac field on an arbitrary lattice.

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        m (float): The mass parameter in the Dirac equation corresponding to the bare particle mass

    Returns:
        FermionicOperator
    """

    gamma0 = rep['gamma0']
    terms = []

    # Sum over the sites
    for site in lattice.sites:
        # And all spinor components
        for alpha in range(2):
            for beta in range(2):
                # Gamma0 gives the coupling between the field components.
                terms.append(
                    m * psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(site, beta, lattice)
                )

    return operator_sum(terms)

################################################################################################
# 2. Set up the gauge energy term
################################################################################################


def hamilton_gauge(lattice, S, e=1.):
    """
    This function contructs the gauge energy term of the `Wilson` Hamiltonian on an arbitrary lattice.

    Args:
        lattice (Lattice): The lattice object
        S (float): Spin truncation value of the Quantum Link model (must be integer or half-integer valued)
        e (float): The charge parameter in the Dirac equation.

    Returns:
        SpinSOperator
    """
    terms = []

    # sum over all links in the lattice
    for edge in lattice.edges:
        # for each link, add a term E^2 to the sum
        terms.append(E2(edge, S, lattice, e))
    # sum up terms
    return operator_sum(terms)

################################################################################################
# 3. Set up the Wilson regulation term
################################################################################################


def hamilton_wilson(lattice, rep=dirac, r=1., a=1.):
    """
    This function contructs the Wilson regulator term of the `Wilson` Hamiltonian for a two component
    dirac field ona a ONE-DIMENSONAL lattice.
    # TODO: Has to be adapted for > 1 dimension.

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        r (float): Should be between 0 and 1. This parameter regulates the strength of the wilson
            regulator term
        a (float): The lattice spacing.

    Returns:
        FermionicOperator
    """

    gamma0 = rep['gamma0']
    summands = []

    # sum over all lattice sites
    for site in lattice.sites:
        prev_site = lattice.project(site - 1)
        next_site = lattice.project(site + 1)
        # Treat the spinor components
        for alpha in range(2):
            for beta in range(2):
                psibar_psiprev = psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(prev_site, beta, lattice)
                psibar_psi = psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(site, beta, lattice)
                psibar_psinext = psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(next_site, beta, lattice)

                second_diff = (psibar_psiprev - 2 * psibar_psi + psibar_psinext)
                # print(second_diff)
                summands.append(second_diff)

    return -r / (2 * a) * operator_sum(summands)


################################################################################################
# 4. Set up the hopping term
################################################################################################

def hamilton_hopping(lattice, rep=dirac, t=1., a=1.):
    """
        This function contructs hopping part of the Hamilltonian for free Wilson fermions in 1d.

        Args:
            lattice (Lattice): The lattice object
            rep (dict): The representation of the Clifford algebra to use.
            t (float): A hopping parameter to regulate the strength of the hopping term.
            a (float): The lattice spacing.

        Returns:
            FermionicOperator
        """

    gamma0 = rep['gamma0']
    gamma1 = rep['gamma1']

    gamma_mix = -1j * gamma0 @ gamma1
    # print('gamma_mix:\n', gamma_mix.data.todense())

    summands = []
    # Sum over all lattice sites (--> needs to go to a sum over edges or so for higher dim)
    for site in lattice.sites:
        next_site = lattice.project(site + 1)

        for alpha in range(2):
            for beta in range(2):
                bwd_hopp = psidag(site, alpha, lattice) * gamma_mix[alpha, beta] * psi(next_site, beta, lattice)
                fwd_hopp = bwd_hopp.dag()

                summands += [fwd_hopp, bwd_hopp]

    return t / (2 * a) * operator_sum(summands)
