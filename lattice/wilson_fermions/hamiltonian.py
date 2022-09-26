from lattice.wilson_fermions.clifford import gamma5
from ..lattice import Lattice
from ..operators.spin_operators import SpinSOperator
from ..operators.fermionic_operators import FermionicOperator
from .gauss_law import *
import numpy as np

################################################################################################
# 1. Set up the mass term
################################################################################################


def hamilton_mass(lattice, rep, params) -> FermionicOperator:
    """
    This function contructs the mass term of the `Wilson` Hamiltonian of a two component
    dirac field on an arbitrary lattice. Works in principle in any lattice dimension if given
    a suitable 2-dim. representation of the relevant Clifford algebra.

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        params (dict): The parameter dictionary for the Hamiltionan.
            Must contain:
                m (float): the mass parameter in the Dirac equation corresponding to the bare particle mass

    Returns:
        mass_term (FermionicOperator)
    """

    # 1. Extract the gamma matrix and relevant parameters
    gamma0 = rep['gamma0']
    m = params['m']

    # 2. Initialize an empty list for the terms
    terms = []

    # 3. Go over all sites and create the mass_terms
    for site in lattice.sites:
        # And all spinor components
        for alpha in range(2):
            for beta in range(2):
                # gamma0 gives the coupling between the field components.
                terms.append(
                    m * psidag(site, alpha, lattice) * gamma0[alpha, beta] * psi(site, beta, lattice)
                )

    # 4. Sum and tensor with an identity on the links if there are links.
    mass_term = operator_sum(terms)
    return mass_term

################################################################################################
# 2. Set up the gauge energy term
################################################################################################


def hamilton_gauge(lattice, params) -> SpinSOperator:
    """
    Contructs the gauge energy term of the `Wilson` Hamiltonian on an arbitrary lattice.
    Works in any lattice dimension.

    Args:
        lattice (Lattice): The lattice object
        params (dict): A dictionary containing the parameters for the Hamiltonian terms.
            Must contain:
                S (float): Spin truncation value of the Quantum Link model (must be integer or half-integer valued)
                e (float): The charge parameter in the Dirac equation.
            Optional:
                theta (float): Topological theta-term corresponding to a constant electric background field.
                    Default value of theta is 0, if not provided.

    Returns:
        SpinSOperator
    """

    # 1. Extract the relevant parameters:
    S = params['S']
    e = params['e']

    # 2. If theta term is nonzero, include it in the energy term, otherwise set it to zero
    if 'theta' in params.keys():
        theta = params['theta']
    else:
        theta = 0.

    # 3. Initialize an empty list to store all gauge energy terms
    terms = []

    # 4. Create squared flux operators E2 for all lattice edges
    for edge in lattice.edges:
        # for each link, add a term E^2 to the sum
        terms.append(E2(edge, S, lattice, e=e, theta=theta))  # note: the factor e^2 is included in E2

    # 5. Sum up terms and return
    return 0.5 * operator_sum(terms)

################################################################################################
# 3. Set up the plaquette term
################################################################################################


def hamilton_plaquette(lattice, params) -> SpinSOperator:
    """
    Construct the plaquette part of the Hamiltonian. Works for arbitrary dimension > 2.

    Args:
        lattice (Lattice): The lattice object for which to calculate the plaquette hamilton_qiskit
        params (dict): A dictionary with parameters for the Hamiltonian.
            Must contain:
                S (float): The spin truncation value of the Quantum Link Model, must be positive integer or half-integer
                e (float): The charge parameter in the Dirac equation

    Returns:
        SpinSOperator
    """
    # 1. Set up an empty list to store all plaquette operators
    summands = []
    S = params['S']
    e = params['e']

    # 2. Iterate plaquettes in the lattice (sites & two dimensions to walk in)
    for site in lattice.sites:
        for mu in range(lattice.ndim):
            for nu in range(mu + 1, lattice.ndim):

                # 2.1 Catch and ignore plaquettes at the boundaries (if lattice is not peridoic)
                plaquette_inexistent = (lattice.is_boundary_along(site, mu, direction='positive') or
                                        lattice.is_boundary_along(site, nu, direction='positive')
                                        )
                if plaquette_inexistent:
                    # print('Plaquette ({}, {}, {}) inexistent'.format(site, mu, nu))
                    continue

                # 2.2 Deal with plaquettes that are in the lattice
                else:
                    # Append the plaquette sum operator 0.5*(U_mu_nu + h.c.) to the summands
                    summands.append(plaquette_sum(site, mu, nu, S, lattice))

    # 3. Sum together all plaquette terms and return the resulting plaquette hamilton_qiskit
    return -1. / (e ** 2) * operator_sum(summands)

################################################################################################
# 4. Set up the hopping term
################################################################################################


def hamilton_hopping(lattice, rep, params, output='qiskit'):
    """
    Constructs the hopping term for the `Wilson` Hamiltonian for a two component
    dirac field ona a 1 or 2 dimensional lattice.
    (In principle works in any dimension, but step 2. needs to be adapted by supplying
    more gamma matrices)

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        params (dict): The dictonary of physical parameters for the Hamiltonian. Must contain:
            S (float): The spin truncation value of the Quantum Link Model for the gauge field Hilbert spaces. Has
                to be a positive integer or half integer or zero for non-interacting Wilson fermions.
            t (float): A strength parameter for the hopping
            a (float): The lattice spacing.
        output(str): The desired output format.
            Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix', 'abstract']

    Returns:
        FermionicOperator or qiskit.aqua.Operator or qutip.Qobj or np.ndarray
    """

    # 1. Get dimensionality of the lattice and extract the physical parameters
    ndim = lattice.ndim
    S = params['S']
    t = params['t']
    a = params['a']

    # 2. Get the dirac matrices for the given representation of the clifford algebra
    if ndim == 1:
        gamma = [rep['gamma0'], rep['gamma1']]
    elif ndim == 2:
        gamma = [rep['gamma0'], rep['gamma1'], rep['gamma2']]
    else:
        raise NotImplementedError("Currently only supports 1 and 2 space-dimensions. Please " +
                                  "provide a lattice of dimension 1 or 2.")

    # 3. Initialize an empty list for the summands
    summands = []

    # 4. Sum over all lattice sites
    for site in lattice.sites:
        # 4.1 Sum over all directions
        for j in range(ndim):
            # Treat open and closed boundary conditions (skip term if edge is at boundary)
            is_at_boundary = lattice.is_boundary_along(site, j, direction='positive')
            if is_at_boundary:
                # print('encountered boundary at {}, {}'.format(site, j))
                continue

            # Get the edge along which hopping takes place and the next site
            next_site = lattice.project(site + standard_basis(j, ndim))
            edge = (lattice.site_index(site), j)
            # print(edge)
            # Construct the product of matrices -1j * gamma0 * gammaj for that direction
            gamma_mix_j = -1j * gamma[0] @ gamma[j + 1]
            # print('gamma_mix_{}:\n'.format(j+1), gamma_mix_j.full())

            # 4.2 Sum over all spinor components:
            for alpha in range(2):
                for beta in range(2):
                    # Skip cases with zero coefficients
                    if gamma_mix_j[alpha, beta] == 0:
                        # print('skipped ab {}, {}'.format(alpha, beta))
                        continue

                    # Generate the fermionic hopping terms
                    hopp_coeff = (t / (2 * a))
                    bwd_hopp = (hopp_coeff
                                * psidag(site, alpha, lattice)
                                * gamma_mix_j[alpha, beta]
                                * psi(next_site, beta, lattice))
                    fwd_hopp = bwd_hopp.dag()

                    # If gauge field is present, tensor with the hopping terms with the link operators
                    if S > 0:
                        bwd_hopp = bwd_hopp @ U(edge, S, lattice)
                        fwd_hopp = fwd_hopp @ Udag(edge, S, lattice)

                    # Convert the operators to qubit operators and add them to the total hopping term
                    if not output == 'abstract':
                        bwd_hopp = bwd_hopp.to_qubit_operator(output=output)
                        fwd_hopp = fwd_hopp.to_qubit_operator(output=output)

                    summands += [fwd_hopp, bwd_hopp]

    # 5. Sum up all hopping terms and return the hopping hamilton_qiskit
    return operator_sum(summands)

################################################################################################
# 5. Set up the wilson regulator term
################################################################################################


def hamilton_wilson(lattice, rep, params, output='qiskit'):
    """
    This function contructs the Wilson regulator term of the `Wilson` Hamiltonian for a two component
    dirac field. Works in any dimension.

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        params (dict): The dictonary of physical parameters for the Hamiltonian. Must contain:
            S (float): The spin truncation value of the Quantum Link Model for the gauge field Hilbert spaces. Has
                to be a positive integer or half integer or zero for non-interacting Wilson fermions.
            r (float): A strength parameter for the Wilson regulator. Should be between 0 and 1.
            a (float): The lattice spacing.
        output(str): The desired output format.
            Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix', 'abstract']

    Returns:
        FermionicOperator or qiskit.aqua.Operator or qutip.Qobj or np.ndarray
        """

    # 1. Get dimensionality of the lattice and the gamma0 matrix and model parameters
    ndim = lattice.ndim
    gamma0 = rep['gamma0']
    S = params['S']
    r = params['r']
    a = params['a']

    # 2. Initialize empty lists to store the summands
    diagonal_summands = []
    hopping_summands = []

    # 3. Sum over all lattice sites
    for site in lattice.sites:

        # 3.1 Constructing the diagonal summands
        for alpha in range(2):
            for beta in range(2):
                # Skip cases with zero coefficients
                if gamma0[alpha, beta] != 0:
                    diagonal_summand = (r * ndim / a) * psidag(site, alpha, lattice) \
                                       * gamma0[alpha, beta] * psi(site, beta, lattice)
                    diagonal_summands.append(diagonal_summand)

        # 3.2 Construct the off-diagonal (hopping) summands
        # Sum over all directions
        for j in range(ndim):
            # Treat open and closed boundary conditions (skip term if edge is at boundary)
            site_is_at_boundary = lattice.is_boundary_along(site, j, direction='positive')
            if site_is_at_boundary:
                # print('encountered boundary at {}, {}'.format(site, j))
                continue

            # Get the edge along which hopping takes place and the next site
            next_site = lattice.project(site + standard_basis(j, ndim))
            edge = (lattice.site_index(site), j)

            # 3.2.1 Sum over all spinor components:
            for alpha in range(2):
                for beta in range(2):
                    # Skip cases with zero coefficients
                    if gamma0[alpha, beta] == 0:
                        # print('skipped ab {}, {}'.format(alpha, beta))
                        continue

                    # Generate the fermionic hopping terms
                    wilson_coeff = -(r / (2 * a))
                    bwd_hopp = (wilson_coeff
                                * psidag(site, alpha, lattice)
                                * gamma0[alpha, beta]
                                * psi(next_site, beta, lattice))
                    fwd_hopp = bwd_hopp.dag()

                    # If gauge field is present, tensor with the hopping terms with the link operators
                    if S > 0:
                        bwd_hopp = bwd_hopp @ U(edge, S, lattice)
                        fwd_hopp = fwd_hopp @ Udag(edge, S, lattice)

                    # Convert the operators to qubit operators and add them to the total hopping term
                    if not output == 'abstract':
                        bwd_hopp = bwd_hopp.to_qubit_operator(output=output)
                        fwd_hopp = fwd_hopp.to_qubit_operator(output=output)

                    hopping_summands += [fwd_hopp, bwd_hopp]

    # 4. finalizing the diagonal term
    diagonal_term = operator_sum(diagonal_summands)
    if S > 0:
        diagonal_term = diagonal_term @ link_id(S, lattice)
    if not output == 'abstract':
        diagonal_term = diagonal_term.to_qubit_operator(output=output)

    # 5. finalizing the hopping term
    hopp_term = operator_sum(hopping_summands)

    # 6. summing hopping and diagonal term together
    return diagonal_term + hopp_term

################################################################################################
# 6. Set up a simplified routine to get the combined mass-wilson-hopping part of the Hamiltonian
################################################################################################

def hamilton_hopp_mass_wilson(lattice,rep,params,output='qiskit'):
    mass, hopp = terms_hamilton_hopp_mass_wilson(lattice,rep,params,output)
    return mass + hopp

def terms_hamilton_hopp_mass_wilson(lattice, rep, params, output='qiskit'):
    """
    Constructs the hopping-mass-wilson part of the Hamiltonian for a two component
    dirac field ona a one or two dimensional lattice.

    Args:
        lattice (Lattice): The lattice object
        rep (dict): The representation of the Clifford algebra to use.
        params (dict): A dictonary of the model parametesr for the Hamiltonian. Must contain
            S (float): The spin truncation value of the Quantum Link Model for the gauge field Hilbert spaces. Has
                to be a positive integer or half integer or zero for non-interacting Wilson fermions.
            t (float): A strength parameter for the hopping
            r (float): Should be between 0 and 1. This parameter regulates the strength of the wilson
                regulator term
            a (float): The lattice spacing.
            m (float): The bare mass of the Wilson fermions (mass parameter in the Hamiltonian)

    Returns:
        FermionicOperator
        """

    # 1. Get dimensionality of the lattice and extract the model parameters
    ndim = lattice.ndim
    S = params['S']
    a = params['a']
    t = params['t']
    r = params['r']
    m = params['m']

    # 2. Get the dirac matrices for the given representation of the clifford algebra
    if ndim == 1:
        gamma = [rep['gamma0'], rep['gamma1']]
    elif ndim == 2:
        gamma = [rep['gamma0'], rep['gamma1'], rep['gamma2']]
    else:
        raise NotImplementedError("Currently only supports 1 and 2 space-dimensions. Please " +
                                  "provide a lattice of dimension 1 or 2.")

    # 3. Initialize empty lists to store the summands
    diagonal_summands = []
    hopping_summands = []

    # 4. Sum over all lattice sites
    for site in lattice.sites:

        # 4.1 Constructing the diagonal summands
        #  Sum over the two components of the two-component spinors
        for alpha in range(2):
            for beta in range(2):
                # Skip cases with zero coefficients
                if gamma[0][alpha, beta] != 0:
                    diagonal_summand = (m + r * ndim / a) \
                                       * psidag(site, alpha, lattice) \
                                       * gamma[0][alpha, beta] * psi(site, beta, lattice)
                    diagonal_summands.append(diagonal_summand)

        # 4.2 Construct the off-diagonal (hopping) summands
        # Sum over all directions
        for j in range(ndim):
            # Treat open and closed boundary conditions (skip term if edge is at boundary)
            site_is_at_boundary = lattice.is_boundary_along(site, j, direction='positive')
            if site_is_at_boundary:
                # print('encountered boundary at {}, {}'.format(site, j))
                continue

            # Get the edge along which hopping takes place and the next site
            next_site = lattice.project(site + standard_basis(j, ndim))
            edge = (lattice.site_index(site), j)
            # Construct the product of matrices -1j * gamma0 * gammaj for that direction
            gamma_mix_j = gamma[0] @ (1j * t * gamma[j + 1] + r * np.eye(2))
            # print('gamma_mix_{}:\n'.format(j+1), gamma_mix_j.full())

            # 4.2.1 Sum over all spinor components:
            for alpha in range(2):
                for beta in range(2):
                    # Skip cases with zero coefficients
                    if gamma_mix_j[alpha, beta] == 0:
                        # print('skipped ab {}, {}'.format(alpha, beta))
                        continue

                    # Generate the fermionic hopping terms
                    coeff = -(1. / (2 * a))
                    bwd_hopp = (coeff
                                * psidag(site, alpha, lattice)
                                * gamma_mix_j[alpha, beta]
                                * psi(next_site, beta, lattice))
                    fwd_hopp = bwd_hopp.dag()

                    # If gauge field is present, tensor with the hopping terms with the link operators
                    if S > 0:
                        bwd_hopp = bwd_hopp @ U(edge, S, lattice)
                        fwd_hopp = fwd_hopp @ Udag(edge, S, lattice)

                    # Convert the operators to qubit operators and add them to the total hopping term
                    if not output == 'abstract':
                        bwd_hopp = bwd_hopp.to_qubit_operator(output=output)
                        fwd_hopp = fwd_hopp.to_qubit_operator(output=output)

                    hopping_summands += [fwd_hopp, bwd_hopp]

    # 5. finalizing the diagonal term
    diagonal_term = operator_sum(diagonal_summands)
    if S > 0:
        diagonal_term = diagonal_term @ link_id(S, lattice)
    if not output == 'abstract':
        diagonal_term = diagonal_term.to_qubit_operator(output=output)

    # 6. finalizing the hopping term
    hopp_term = operator_sum(hopping_summands)

    # 7. summing hopping and diagonal term together
    return diagonal_term, hopp_term


################################################################################################
# 7. Combine all routines to build up the total hamilton_qiskit
################################################################################################


def build_hamilton(lattice, params, rep=dirac, lam=20., boundary_cond=None, output='qutip'):
    """
    Wrapper to build the full Hamiltonian of a multidimensional Wilson fermion U(1) LGT

    Args:
        lattice (Lattice): The lattice on which the model is built
        params (dict): The dictionary of physical parameters of the model
            Must contain
            S (float): The spin truncation value of the Quantum Link Model for the gauge field Hilbert spaces. Has
                to be a positive integer or half integer or zero for non-interacting Wilson fermions.
            t (float): A strength parameter for the hopping
            r (float): Should be between 0 and 1. This parameter regulates the strength of the wilson
                regulator term
            a (float): The lattice spacing.
            m (float): The bare mass of the Wilson fermions (mass parameter in the Hamiltonian)
        rep (dict): The representation of the Clifford algebra (the gamma matrices) to be used.
        lam (float): The strength of the effective gauge invariance regulator. Must be positive.
        boundary_cond (dict): A dictionary containing the boundary conditions is `lattice` is a Lattice object
            with finite (non-periodic) boundary conditions.
        output (str): The desired output format

    Returns:
        qiskit.aqua.Operator or qutip.Qobj or np.ndarray or scipy.spmatrix

    """

    # print('##### Building the Hamiltonian #########')
    mass_hopping_wilson_part = hamilton_hopp_mass_wilson(lattice,
                                                         rep=rep,
                                                         params=params,
                                                         output=output)

    hamilton = mass_hopping_wilson_part
    # print('Mass, Hopping & Wilson energy added........')
    # print(hamilton)
    if params['S'] > 0:
        # Set up the gauge field energy#nota che deve avere la stessa dimensionlità
        gauge_part = (fermion_id(lattice)
                      @ hamilton_gauge(lattice, params=params)).to_qubit_operator(output=output)

        # Set up the gauge regularization part
        gauge_regularization = gauss_law_regularizer(lattice,
                                                     params,
                                                     lam=lam,
                                                     boundary_cond=boundary_cond,
                                                     output=output)

        hamilton += gauge_part + gauge_regularization
        # print('Gauge field flux energy added.')
        # print('Gauge invariance regulator added.')

        # Set up and add the gauge field plaquette energy if space dimension is > 1.
        if lattice.ndim > 1:
            plaquette_part = (fermion_id(lattice)
                              @ hamilton_plaquette(lattice, params=params)).to_qubit_operator(output=output)

            hamilton += plaquette_part
            # print('Plaquette energy added.')

    # print('##### Hamiltonian successfully built #####')
    return hamilton
