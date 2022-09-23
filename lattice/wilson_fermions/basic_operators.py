#import sys 
#sys.path.append("../")
from ..operators.fermionic_operators import BaseFermionOperator
from ..operators.spin_operators import BaseSpinOperator
from ..operators.qiskit_aqua_operator_utils import operator_sum
import numpy as np
import scipy

################################################################################################
# 0. Set up helper functions

################################################################################################


def standard_basis(i, n):
    """
    Constructs the basis vector e_i of the Euclidean standard basis in a vectorspace of dim `n`.
    E.g. standard_basis(2,3) = [[0., 0., 1., 0.]]

    Args:
        i(int): The index of the basis vector. Return array has a 1 at the i-th position.
        n(int): Length of the return array. Corresponds to the dimensionality of the vectorspace.

    Returns:
        np.ndarray
    """
    assert 0 <= i < n, 'i must be between in [0,n)'
    vec = np.zeros(n, dtype=int)
    vec[i] = 1
    return vec

################################################################################################
# 1. Set up the basic fermionic annihilation and creation operators
################################################################################################


def psi(site, component, lattice, ncomponents=2):
    """The fermionic annihilation operator for the field component `component` at lattice site `site`.
    Note: This method works for arbitrary lattice sizes and geometries.
    """
    # get the index of `site`
    site_index = lattice.site_index(site)

    # Total number of fermionic operators (assuming 2 components - 2dim. rep. of the clifford algebra CL(1,1)
    n_ops_total = lattice.nsites * ncomponents
    n_ops_before = ncomponents * site_index + component

    return BaseFermionOperator('I' * n_ops_before + '-' + 'I' * (n_ops_total - n_ops_before - 1))


def psidag(site, component, lattice, ncomponents=2):
    """The fermionic creation operator for the field component `component` at lattice site `site` """
    return psi(site, component, lattice, ncomponents).dag()


def fermion_id(lattice, ncomponents=2):
    """The identity operator on the fermionic/site degrees of freedom"""
    identity = BaseFermionOperator('I' * lattice.nsites * ncomponents)
    return identity

################################################################################################
# 2. Set up the basic gauge field operators
################################################################################################

# 2.1 Calculate the local embeddings to speed up the .to_qubit_operator() conversions

padding_value = 1
print("padding_value= ", padding_value )
spin05_trafo = BaseSpinOperator(0.5,[0],[0],[0])._logarithmic_encoding()
spin1_trafo = BaseSpinOperator(1,[0],[0],[0])._logarithmic_encoding(embed_padding=padding_value)
spin15_trafo = BaseSpinOperator(1.5,[0],[0],[0])._logarithmic_encoding(embed_padding=padding_value)
spin2_trafo = BaseSpinOperator(2.0,[0],[0],[0])._logarithmic_encoding(embed_padding=padding_value)

#spin05_trafo = BaseSpinOperator(0.5,[0],[0],[0])._linear_encoding()
#spin1_trafo = BaseSpinOperator(1,[0],[0],[0])._linear_encoding()
#spin15_trafo = BaseSpinOperator(1.5,[0],[0],[0])._linear_encoding()
#spin2_trafo = BaseSpinOperator(2.0,[0],[0],[0])._linear_encoding()

local_embedding = {
    0.5 : spin05_trafo,
    1.0 : spin1_trafo,
    1.5 : spin15_trafo,
    2.0 : spin2_trafo
}

# 2.2 Define the operators
def U(edge, S, lattice):
    """
    Creates the Quantum Link Model flux raising operator 'U'

    Args:
        edge (array-like):
            The lattice edge on which the flux raising field operator acts on. Edge should be
            in the format [site_index, direction].
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        S (float):
            The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice):
            The lattice object

    Returns:
        SpinSOperator object
    """

    # 1. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')

    # 2. Construct and return the operator
    index = lattice.edge_index(edge)
    N = lattice.nedges

    position_marker = [0] * index + [1] + [0] * (N - index - 1)

    Sx = BaseSpinOperator(S, Sx=position_marker, Sy=[0] * N, Sz=[0] * N)
    Sy = BaseSpinOperator(S, Sx=[0] * N, Sy=position_marker, Sz=[0] * N)

    # 3. Pre-set the XYZI local embedding transformation (for faster transform later)
    Sx.transformed_XYZI = local_embedding[S]
    Sy.transformed_XYZI = local_embedding[S]

    norm_factor = 1. / np.sqrt( (S+1) * S )
    return norm_factor * (Sx + 1j * Sy)


def Udag(edge, S, lattice):
    """
    Creates the Quantum Link Model lowering operator 'U^dagger'

    Args:
        edge (array-like):
            The lattice edge on which the lowering field operator acts on. Edge should be
            in the format [site_index, direction].
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        S (float):
            The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice):
            The lattice object

    Returns:
        SpinSOperator object
    """
    return U(edge, S, lattice).dag()


def E(edge, S, lattice, e=1., theta=0.):
    """
    Creates the gauge flux operator 'E'

    Args:
        edge (array-like):
            The lattice edge on which the electric field operator acts on. Edge should be
            in the format [site_index, direction].
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        S (float):
            The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice):
            The lattice object
        e (float): optional
            The parameter value for the unit of charge from the physical model
        theta (float): optional,
            The theta vacuum angle. A non-zero theta angle adds a constant background electric field to the model.

    Returns:
        SpinSOperator object
    """

    # 1. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')

    # 2. Find the positions of the relevant spin systems in the register
    index = lattice.edge_index(edge)
    N = lattice.nedges
    position_marker = [0] * index + [1] + [0] * (N - index - 1)
    # 3. Construct the operator
    Sz = BaseSpinOperator(S, Sx=[0] * N, Sy=[0] * N, Sz=position_marker)
    # 4. Initialize the transform with a pre-computed transform (to speed up the to-qubit-transform)
    Sz.transformed_XYZI = local_embedding[S]
    # 5. Return the operator E (plus an additional theta term offset, if requested)
    if abs(theta) > 0:
        return e * (Sz + theta * link_id(S, lattice))
    else:
        return e * Sz


def E2(edge, S, lattice, e=1., theta=0.):
    """
    Creates the square 'E^2' of gauge flux operator 'E'

    Args:
        edge (array-like):
            The lattice edge on which the electric field operator acts on. Edge should be
            in the format [site_index, direction].
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        S (float):
            The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice):
            The lattice object
        e (float): optional
            The parameter value for the unit of charge from the physical model
        theta (float): optional,
            The theta vacuum angle. A non-zero theta angle adds a constant background electric field to the model.

    Returns:
        SpinSOperator object
    """

    # 1. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')

    # 2. Find the positions of the relevant spin systems in the register
    index = lattice.edge_index(edge)
    N = lattice.nedges

    # 2.1 Set the order of the spin Z operator at the given index to 2
    position_marker = [0] * index + [2] + [0] * (N - index - 1)
    # 3. Construct the operator
    Sz2 = BaseSpinOperator(S, Sx=[0] * N, Sy=[0] * N, Sz=position_marker)
    # 4. Initialize the transform with a pre-computed transform (to speed up the to-qubit-transform)
    Sz2.transformed_XYZI = local_embedding[S]
    # 5. Return the operator E^2 (plus an additional theta term offset, if requested)
    if abs(theta) > 0:
        electric_term = e**2 * Sz2
        mixed_term = 2 * (e * theta) * E(edge, S, lattice, e=e, theta=0.)
        theta_term = (e * theta) ** 2 * link_id(S, lattice)
        return electric_term + mixed_term + theta_term

    else:
        return e**2 * Sz2


def link_id(S, lattice):
    """The identity operator on the gauge degrees of freedom

    Args:
        S (float):
            The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice):
            The lattice object

    Returns:
        SpinSOperator object
    """

    # 1. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')

    # 2. Construct and return the operator
    identity = BaseSpinOperator(S, [0] * lattice.nedges, [0] * lattice.nedges, [0] * lattice.nedges)
    identity.transformed_XYZI = local_embedding[S]
    return identity


def plaquette(site, mu, nu, S, lattice, output='qiskit'):
    """
        Constructs the plaquette operator U_mu_nu(site)

    Args:
        site (np.ndarray): The lattice site from which the plaquette directions are counted.
        mu (int): The first direction of the plaquette. Must be in [0, lattice.ndim)
        nu (int): The second direction of the plaquette. Must be different from `mu` and in [0, lattice.ndim)
        S (float): The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice): The lattice object
        output (str): The desired output type. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']

    Returns:
        qiskit.aqua.Operator or qutip.Qobj or np.ndarray
    """

    # 0. Parsing:
    assert mu in np.arange(lattice.ndim), "mu={} not in lattice with dimenson ndim={}".format(mu, lattice.ndim)
    assert nu in np.arange(lattice.ndim), "nu={} not in lattice with dimenson ndim={}".format(mu, lattice.ndim)
    assert mu != nu, 'mu and nu cannot be equal.'

    # 1. Generating the sites along the required dimensions `mu` and `nu`
    site = np.asarray(site)
    site_plus_mu = lattice.project(site + standard_basis(mu, lattice.ndim))
    site_plus_nu = lattice.project(site + standard_basis(nu, lattice.ndim))

    # 2. Extracting the site indices from the site vectors
    x_index = lattice.site_index(site)
    xplusmu_index = lattice.site_index(site_plus_mu)
    xplusnu_index = lattice.site_index(site_plus_nu)

    # 3. Combining the site indices and directions to the respective edges in the lattice
    edge1 = (x_index, mu)
    edge2 = (xplusmu_index, nu)
    edge3 = (xplusnu_index, mu)
    edge4 = (x_index, nu)

    # 4. Setting up the relevant edge operators for the plaquette product
    U_edge1 = U(edge1, S, lattice).to_qubit_operator(output=output)
    U_edge2 = U(edge2, S, lattice).to_qubit_operator(output=output)
    U_edge3_dag = Udag(edge3, S, lattice).to_qubit_operator(output=output)
    U_edge4_dag = Udag(edge4, S, lattice).to_qubit_operator(output=output)

    # 5. Multiplying to plaquette and returning
    return U_edge1 * U_edge2 * U_edge3_dag * U_edge4_dag


def plaquette_sum(site, mu, nu, S, lattice):
    """
    Constructs the plaquette sum operator 0.5*(U_mu_nu(site) + U_mu_nu(site)^dagger) used in the 
    construction of the plaquette term in the Hamiltonian.

    Args:
        site (np.ndarray): The lattice site from which the plaquette directions are counted.
        mu (int): The first direction of the plaquette. Must be in [0, lattice.ndim)
        nu (int): The second direction of the plaquette. Must be different from `mu` and in [0, lattice.ndim)
        S (float): The spin truncation parameter of the Quantum Link Model. Must be a positive integer or half-int.
        lattice (Lattice): The lattice object

    Returns:
        SpinSOperator
    """

    # 0. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')
    #    Parse valid directions
    assert mu in np.arange(lattice.ndim), "mu={} not in lattice with dimenson ndim={}".format(mu, lattice.ndim)
    assert nu in np.arange(lattice.ndim), "nu={} not in lattice with dimenson ndim={}".format(mu, lattice.ndim)
    assert mu != nu, "mu and nu cannot be equal"

    # 1. Generating the sites along the requried dimensions `mu` and `nu`
    site = np.asarray(site)
    site_plus_mu = lattice.project(site + standard_basis(mu, lattice.ndim))
    site_plus_nu = lattice.project(site + standard_basis(nu, lattice.ndim))

    # 2. Extracting the site indices from the site vectors
    x_index = lattice.site_index(site)
    x_mu_index = lattice.site_index(site_plus_mu)
    x_nu_index = lattice.site_index(site_plus_nu)

    # 3. Combining the site indices and directions to the respective edges in the lattice
    edge1 = (x_index, mu)
    edge2 = (x_mu_index, nu)
    edge3 = (x_nu_index, mu)
    edge4 = (x_index, nu)
    edges = [edge1, edge2, edge3, edge4]

    # 4. Mark the spin register position of the four relevant edges:
    N = lattice.nedges
    #  Get the indices of the edges (corresponds to ordering of the spin register)
    edge_indices = [lattice.edge_index(edge) for edge in edges]
    #  Create 4 arrays of length N, one for each edge, in which only the position corresponding to 
    #  the specific edge is 1 all others 0 (e.g. [0,1,0,0] for edge with index 2 on a lattice with 4 edges)
    pos_markers = [standard_basis(index, N) for index in edge_indices]

    # 5. Set up the 8 relevant terms for 0.5*(U_mu_nu + U_mu_nu^dagger)
    xxxx = BaseSpinOperator(S, Sx=sum(pos_markers), Sy=[0] * N, Sz=[0] * N)
    yyyy = BaseSpinOperator(S, Sx=[0] * N, Sy=sum(pos_markers), Sz=[0] * N)

    xxyy = BaseSpinOperator(S, Sx=pos_markers[0] + pos_markers[1], Sy=pos_markers[2] + pos_markers[3], Sz=[0] * N)
    yyxx = BaseSpinOperator(S, Sx=pos_markers[2] + pos_markers[3], Sy=pos_markers[0] + pos_markers[1], Sz=[0] * N)

    yxyx = BaseSpinOperator(S, Sx=pos_markers[1] + pos_markers[3], Sy=pos_markers[0] + pos_markers[2], Sz=[0] * N)
    xyxy = BaseSpinOperator(S, Sx=pos_markers[0] + pos_markers[2], Sy=pos_markers[1] + pos_markers[3], Sz=[0] * N)

    xyyx = BaseSpinOperator(S, Sx=pos_markers[0] + pos_markers[3], Sy=pos_markers[1] + pos_markers[2], Sz=[0] * N)
    yxxy = BaseSpinOperator(S, Sx=pos_markers[1] + pos_markers[2], Sy=pos_markers[0] + pos_markers[3], Sz=[0] * N)

    summands = [xxxx, yyyy, -xxyy, -yyxx, xyxy, yxyx, xyyx, yxxy]
    # 5.1 Initialize the transform for the operator-to-qubit transformation:
    for summand in summands:
        summand.transformed_XYZI = local_embedding[S]

    # 6. Sum the 8 relevant terms to the sum: 0.5*(U_mu_nu + U_mu_nu^dagger) and return
    return 1. / (S * (S + 1)) ** 2 * operator_sum(summands)
