import numpy as np
from ..operators.fermionic_operators import FermionicOperator, BaseFermionOperator
from ..wilson_fermions.basic_operators import psi, psidag, fermion_id
from ..wilson_fermions.clifford import *


################################################################################################
# 0. Set up group generators for the considered gauge group

###############################################################################################

def elementary_matrix(i, j, n):
    """
        Constructs the elementary matrix E_{i,j} of dimensions (n x n), where all its elements are 0 except when
        (row,column) = (i,j), its element is 1.
        E.g. elementary_matrix(1,0,3) = [[0., 0., 0.]
                                         [1., 0., 0.]
                                         [0., 0., 0.]]

        Args:
            i (int): The row index of the elementary matrix.
            j (int): The column index of the elemewntary.
            n (int): Dimension of the return matrix.
                Return matrix has a 1 at position (row,column) = (i,j)

        Returns:
            np.ndarray
    """
    assert 0 <= i < n and 0 <= j < n, "i and j must be between in [0,N)"
    matrix = np.zeros((n, n))
    matrix[i][j] = 1
    return matrix


def get_generators_SU(n, rep='fundamental'):
    """
        Construct the group generators of the special unitary group SU(n) with normalization
        Tr( T_i * T_j) = 0.5 delta_ij .

        Args:
            n (int): The degree of the special unitary group SU(n)
            rep (str): The representation of the of the group
                rep must be one of the supported representations ['fundamental', 'adjoint']

        Returns:
            list
                The list of all the generators T_2 (np.ndarray), where dim(SU(n)) = n**2 - 1
                [ T_1, T_2, ..., T_{dim(SU(n))} ]
    """
    assert isinstance(n, (int, np.integer)) and n > 0, 'The degree n of the group SU(n) must be a positive integer'

    generators = []
    ngen = n ** 2 - 1

    if rep == 'fundamental':

        # 1. construct the symmetric generators (number of such generators = n*(n-1)/2 )
        for k in range(n):
            for l in range(k):
                gen = elementary_matrix(k, l, n) + elementary_matrix(l, k, n)
                generators.append(gen * 0.5)

        # 2. construct the anti-symmetric generators (number of such generators = n*(n-1)/2 )
        for k in range(n):
            for l in range(k):
                gen = 1.j * (elementary_matrix(k, l, n) - elementary_matrix(l, k, n))
                generators.append(gen * 0.5)

        # 3. construct the remaining diagonal traceless generators (number of such generators = (n-1) )
        for k in range(n-1):
            factor = np.sqrt(2./(k+1+(k+1)**2))
            gen = factor * np.eye(n, n)
            gen[k+1][k+1] = - (k+1) * factor
            for l in range(k+2, n):
                gen[l][l] = 0.
            generators.append(gen * 0.5)

        return generators

    elif rep == 'adjoint':
        # the ((n**2 - 1) x (n**2 - 1)) dimensional generators T_k in the adjoint representation are described by
        # its matrix elements (T_k)_{i,j} = 1.j * f^{kij} where f^{kij} are the structure constants.
        #  [T_i, T_j] = 1.j * f^{ijk} * T_k   and   Tr( T_i * T_j) = 0.5 \delta_ij

        # general version: requires computation of all commutators and then the scalarproduct (given by the trace)
        #                   in order to compute the structure constants. Then the matrix elements can be adjusted.

        # create a list of n**2 - 1 generators
        for k in range(ngen):
            generators.append(np.zeros((ngen, ngen), dtype=complex))

        # Get the generators in the fundamental representation
        fund_gen = get_generators_SU(n)

        # Set up commutator
        def commutator(A, B):
            return A @ B - B @ A

        for i in range(ngen):
            for j in range(i+1, ngen):
                for k in range(j+1, ngen):
                    # get coeff = 1.j * f^{ijk}
                    coeff = 2. * np.trace(commutator(fund_gen[i], fund_gen[j]) @ fund_gen[k])
                    # exploit the total antisymmetry of the structure constants
                    generators[i][j][k] = generators[j][k][i] = generators[k][i][j] = coeff
                    generators[i][k][j] = generators[j][i][k] = generators[k][j][i] = -coeff

        return generators
    else:
        raise UserWarning("`rep` must be one of ['fundamental', 'adjoint']")


################################################################################################
# 1. Set up the basic fermionic annihilation and creation operators
#       with additional color (i.e., group representation) components
###############################################################################################

def psi_color(site, color_component, spinor_component, lattice, ncolors, ncomponents=2):
    """The fermionic annihilation operator for the field spinor component `component` and color
        component `color_component` at lattice site `site`.
        Note: This method works for arbitrary lattice sizes and geometries.
    """
    # Total number of fermionic operators (assuming 2 components - 2dim. rep. of the clifford algebra CL(1,1))
    n_ops_total = lattice.nsites * ncomponents * ncolors
    n_ops_before = lattice.nsites * ncomponents * color_component
    spinor_label = psi(site, spinor_component, lattice, ncomponents).label
    spinor_length = len(spinor_label)

    return BaseFermionOperator('I' * n_ops_before + spinor_label + 'I' * (n_ops_total - n_ops_before - spinor_length))


def psidag_color(site, color_component, spinor_component, lattice, ncolors, ncomponents=2):
    """The fermionic creation operator for the field component `component` at lattice site `site` """
    return psi_color(site, color_component, spinor_component, lattice, ncolors, ncomponents).dag()


def fermion_id_color(lattice, ncolors, ncomponents=2):
    """The identity operator on the fermionic/site degrees of freedom"""
    identity = BaseFermionOperator('I' * lattice.nsites * ncolors * ncomponents)
    return identity


