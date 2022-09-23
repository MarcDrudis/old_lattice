import numpy as np

################################################################################################
# 1. Set up functionality to check for a valid representation of the clifford algebra CL(1,1)
################################################################################################
__all__ = ['zache', 'dirac', 'weyl', 'dirac_prime']


def gamma5(rep, spacetime_dim):
    """Returns the gamma5 matrix in the given 2D-represenatation of the clifford algebra CL(1,1) / CL(1,2)"""
    if spacetime_dim == 2:
        return rep['gamma0'] @ rep['gamma1']
    elif spacetime_dim == 3:
        raise UserWarning('No gamma5 in odd spacetime dimensions.')
    else:
        raise UserWarning('Only supports `spacetime_dim` 2 and 3.')


def check_clifford(representation, signature=(1, -1)):
    """
    Checks if the given representation fulfills the Cliffod algebra for the given `signature` of the metric

    Args:
        representation (dict): A dictionary containing the gamma matrices of a representation of the Clifford
            algebra with the keys 'gamma0', 'gamma1', etc.
        signature (tuple): A tuple specifying the signature of spacetime to be used. E.g. for 2-dimensional Minkowski
            spacetime: (1, -1).

    Returns:
        (bool): True iff the given `representation` is indeed a valid representation of the Clifford algebra for the
            given `signature` of spacetime.
    """

    # 1. Extract the spacetime dimensionality and the metric from the metric signature
    spacetime_dim = len(signature)
    metric = np.diag(signature)
    id = np.eye(2)
    # Set up anticommutator
    anticommutator = lambda A, B: A @ B + B @ A

    # 2. Extract the gamma matrices from the representation of the Clifford algebra
    #    The spacetime dimension determines the number of gamma matrices to extract.
    gamma0 = representation['gamma0']
    gamma1 = representation['gamma1']
    if spacetime_dim== 3:
        # Case Cl(1, 2)
        gamma2 = representation['gamma2']
        gammas = [gamma0, gamma1, gamma2]
    elif spacetime_dim == 2:
        # Case CL(1, 1)
        gammas = [gamma0, gamma1]
    else:
        raise UserWarning('Only supports spacetime dimensions 2 and 3. Make sure your `signature` has length 2 or 3.')

    # 3. Check the anti commutation relations that define the clifford algebra
    for i, gamma_i in enumerate(gammas):
        for j, gamma_j in enumerate(gammas):
            assert np.array_equal(anticommutator(gamma_i, gamma_j), 2 * metric[i, j] * id), \
                'Conflict with relation for (i,j)=({},{})'.format(i, j)

    # 4. Check properties of the defined gamma5 matrix
    if spacetime_dim == 2:
        g5 = gamma5(representation, spacetime_dim=spacetime_dim)
        assert np.array_equal(g5.T.conj(), g5)
        assert np.array_equal(g5 @ g5, id)
        for gamma_i in gammas:
            assert np.array_equal(anticommutator(gamma_i, g5), 0*id)
            assert np.array_equal((g5 @ gamma_i).T.conj(), gamma0 @ g5 @ gamma_i @ gamma0)

    # 5. If all tests succeded, return True.
    return True

################################################################################################
# 2. Define some useful representations of the clifford algebra CL(1,1)
################################################################################################


sigmax = np.array([[0.+0.j, 1.+0.j],
                   [1.+0.j, 0.+0.j]])

sigmay = np.array([[0.+0.j, 0.-1.j],
                   [0.+1.j, 0.+0.j]])

sigmaz = np.array([[1.+0.j,  0.+0.j],
                   [0.+0.j, -1.+0.j]])

zache = {
    'gamma0': sigmax,
    'gamma1': sigmaz*1j,
    'gamma2': sigmay*1j
}

dirac = {
    'gamma0': sigmaz,
    'gamma1': sigmax*1j,
    'gamma2': sigmay*1j
}

dirac_prime = {
    'gamma0': sigmaz,
    'gamma1': sigmay*1j,
    'gamma2': sigmax*1j
}

weyl = {
    'gamma0': sigmax,
    'gamma1': sigmay*1j,
    'gamma2': sigmaz*1j
}

################################################################################################
# 3. Check that they actually satisfy the Clifford algebra relations
################################################################################################

assert check_clifford(weyl)
assert check_clifford(zache)
assert check_clifford(dirac)
assert check_clifford(dirac_prime)