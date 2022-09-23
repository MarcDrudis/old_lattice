import numpy as np
from ..operators.fermionic_operators import FermionicOperator, BaseFermionOperator
from .basic_sun_operators import *
from ..wilson_fermions.hamiltonian import *


################################################################################################
# 1. Set up basic components for the string Hamiltonian
################################################################################################


def conserved_charge(site, generator, lattice, ncomponents=2):
    """
    Generate the conserved charges Q_{site} = psidag_{site} * T * psi_{site}, where T is one generator
    of the gauge group. The conserved charges will enter the longrange interaction term.

    Args:
        site (np.ndarray): The current position on the lattice
        generator (np.ndarray): One generator of the gauge group
        lattice (Lattice): The lattice object
        ncomponents (int): number of spinor components

    Returns:
        FermionicOperator
    """
    # Get total number of colors (given by the dimension of the representation of the gauge group)
    ncolors = generator.shape[0]

    # Calculate the summands of the conserved charge operator
    charge_summands = []
    for spinor_component in range(ncomponents):
        for alpha in range(ncolors):
            for beta in range(ncolors):
                summand = psidag_color(site, alpha, spinor_component, lattice, ncolors, ncomponents) * \
                          generator[alpha][beta] * \
                          psi_color(site, beta, spinor_component, lattice, ncolors, ncomponents)
                charge_summands.append(summand)

    charge = operator_sum(charge_summands)
    return charge


##########################################################################################################
# 2. Set up the Hamiltonian for SU(n) gauge theories in (1+1) spacetime dimensions (wilson & staggered
##########################################################################################################


def string_hamilton_wilson(lattice, group_dim, rep, params, output='abstract'):
    """
        Constructs the `Wilson` Hamiltonian for for a non-abelian SU(n) gauge theory and
        a two component spinor field ona a 1 dimensional lattice.
        (With some minor changes, arbitrary compact continuous gauge groups can be simulated)

        Args:
            lattice (Lattice): The lattice object (must be of dimension 1)
            group_dim (int): The degree `n` of the special unitary group SU(n)
            rep (dict): The representation of the Clifford algebra to use.
            params (dict): The dictonary of physical parameters for the Hamiltonian. Must contain:
                t (float): A strength parameter for the hopping
                r (float): Should be between 0 and 1. This parameter regulates the strength of the wilson
                    regulator term
                a (float): The lattice spacing.
                m (float): The bare mass of the Wilson fermions (mass parameter in the Hamiltonian)
                g (float): The coupling constant
            output (str): The desired output format.
                Must be one of ['abstract', 'qiskit', 'qutip', 'matrix', 'spmatrix']

        Returns:
            FermionicOperator or qiskit.aqua.Operator or qutip.Qobj or np.ndarray
    """

    assert lattice.ndim == 1, "The lattice object must be of dimension 1"
    assert lattice.boundary_cond == 'closed', "The Lattice boundary conditions must be 'closed'"
    assert isinstance(group_dim, (int, np.integer)) and group_dim > 0, 'The degree `group_dim` of the group SU(n) ' \
                                                                       'must be a positive integer'
    # 1. Extract the relevant parameters
    a = params['a']
    g = params['g']
    params.update({'S': 0})

    ncolors = group_dim
    nsites = lattice.nsites

    # 2. Build the Wilson mass terms and the hopping terms with added color indices on the operators psi, psidag
    mass_hopping_terms = hamilton_hopp_mass_wilson(lattice, rep=rep, params=params, output='abstract')
    color_mass_hopping_terms = []
    n_ops_total = nsites * 2 * ncolors
    for color_idx in range(ncolors):
        for summand in mass_hopping_terms.operator_list:
            n_ops_before = nsites * 2 * color_idx
            new_summand = BaseFermionOperator('I' * n_ops_before + summand.label +
                                              'I' * (n_ops_total - n_ops_before - len(summand)), coeff=summand.coeff)
            color_mass_hopping_terms.append(new_summand)

    # 2.1 Finalizing the color_mass_hopping_terms
    color_mass_hopping = operator_sum(color_mass_hopping_terms)
    if not output == 'abstract':
        color_mass_hopping = color_mass_hopping.to_qubit_operator(output=output)

    # 3. Build the longrange interaction terms

    # 3.1 Get a list of the generators of the gauge group SU(n) in the fundamental representation
    generators = get_generators_SU(ncolors)
    ngenerators = len(generators)

    # 3.2 Construct the summands
    longrange_terms = []
    for site in lattice.sites:

        # Treat closed boundary conditions (skip term if edge is at boundary)
        site_is_at_boundary = lattice.is_boundary_along(site, 0, direction='positive')
        if site_is_at_boundary:
            continue

        # Sum over the conserved charges
        for gen_idx in range(ngenerators):
            longrange_inner = []
            for inner_site in lattice.sites:

                # the summation index `long_site` goes from 0 to `site`
                if inner_site[0] > site[0]:
                    break

                inner_summand = conserved_charge(inner_site, generators[gen_idx], lattice)
                longrange_inner.append(inner_summand)

            longrange_summand = (operator_sum(longrange_inner)) ** 2
            longrange_terms.append(longrange_summand)

    # 3.3 Finalizing the longrange_terms
    longrange = g ** 2 * a / 2. * operator_sum(longrange_terms)
    if not output == 'abstract':
        longrange = longrange.to_qubit_operator(output=output)

    return color_mass_hopping + longrange


def string_hamilton_staggered(lattice, group_dim, params, output='abstract'):
    """
        Constructs the `Staggered` Hamiltonian for for a non-abelian SU(n) gauge theory and
        a one component spinor field ona a 1 dimensional lattice.
        (With some minor changes, arbitrary compact continuous gauge groups can be simulated)

        Args:
            lattice (Lattice): The lattice object (must be of dimension 1)
            group_dim (int): The degree of the special unitary group SU(n)
            params (dict): The dictonary of physical parameters for the Hamiltonian. Must contain:
                t (float): A strength parameter for the hopping
                a (float): The lattice spacing.
                m (float): The bare mass of the Wilson fermions (mass parameter in the Hamiltonian)
                g (float): The coupling constant
            output(str): The desired output format.
                Must be one of ['abstract', 'qiskit', 'qutip', 'matrix', 'spmatrix']

        Returns:
            FermionicOperator or qiskit.aqua.Operator or qutip.Qobj or np.ndarray
    """
    assert lattice.ndim == 1, "The lattice object must be of dimension 1"
    assert lattice.boundary_cond == 'closed', "The Lattice boundary conditions must be 'closed'"
    assert isinstance(group_dim, (int, np.integer)) and group_dim > 0, 'The degree `group_dim` of the group SU(n) ' \
                                                                       'must be a positive integer'
    # 1. Extract the relevant parameters
    t = params['t']
    a = params['a']
    m = params['m']
    g = params['g']

    ncolors = group_dim

    # 2. Build the mass term
    mass_terms = []
    for color_idx in range(ncolors):
        for site in lattice.sites:
            # define the index for the phase depending on the summation index `n`
            n = site[0] + 1

            mass_summand = (-1) ** n * psidag_color(site, color_idx, ncolors=ncolors, lattice=lattice,
                                                    spinor_component=0, ncomponents=1) \
                           * psi_color(site, color_idx, ncolors=ncolors, lattice=lattice,
                                       spinor_component=0, ncomponents=1)
            mass_terms.append(mass_summand)

    # 2.1 Finalizing the mass_terms
    mass = m * operator_sum(mass_terms)
    if not output == 'abstract':
        mass = mass.to_qubit_operator(output=output)

    # 3. Build the hopping term
    hopping_terms = []
    for color_idx in range(ncolors):
        for site in lattice.sites:
            # Treat closed boundary conditions (skip term if edge is at boundary)
            site_is_at_boundary = lattice.is_boundary_along(site, 0, direction='positive')
            if site_is_at_boundary:
                continue

            next_site = lattice.project(site + 1)
            hopping_summand = psidag_color(site, color_idx, ncolors=ncolors, lattice=lattice,
                                           spinor_component=0, ncomponents=1) \
                              * psi_color(next_site, color_idx, ncolors=ncolors, lattice=lattice,
                                          spinor_component=0, ncomponents=1)
            hopping_terms.append(hopping_summand)
            hopping_terms.append(hopping_summand.dag())

    # 3.1 Finalizing the mass_terms
    hopping = t / (2*a) * operator_sum(hopping_terms)
    if not output == 'abstract':
        hopping = hopping.to_qubit_operator(output=output)

    # 4. Build the longrange-interaction term
    # 4.1 Get a list of the generators of the gauge group SU(n) in the fundamental representation
    generators = get_generators_SU(ncolors)
    ngenerators = len(generators)

    # 4.2 Construct the summands
    longrange_terms = []
    for site in lattice.sites:

        # Treat closed boundary conditions (skip term if edge is at boundary)
        site_is_at_boundary = lattice.is_boundary_along(site, 0, direction='positive')
        if site_is_at_boundary:
            continue

        # Sum over the conserved charges
        for gen_idx in range(ngenerators):
            longrange_inner = []
            for inner_site in lattice.sites:

                # the summation index `long_site` goes from 0 to `site`
                if inner_site[0] > site[0]:
                    break

                inner_summand = conserved_charge(inner_site, generators[gen_idx], lattice, ncomponents=1)
                longrange_inner.append(inner_summand)

            longrange_summand = (operator_sum(longrange_inner)) ** 2
            longrange_terms.append(longrange_summand)

    # 4.3 Finalizing the longrange_terms
    longrange = g ** 2 * a / 2. * operator_sum(longrange_terms)
    if not output == 'abstract':
        longrange = longrange.to_qubit_operator(output=output)

    return mass + hopping + longrange
