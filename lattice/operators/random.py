from .spin_operators import BaseSpinOperator
from .fermionic_operators import BaseFermionOperator
import numpy.random as rand
import random

def random_spin_operator(S, register_length):
    # Generate random strings of XYZ operators
    Sx, Sy, Sz = [rand.poisson(0.1, size=register_length) for i in range(3)]
    # Generate random coefficient
    coeff = round(rand.normal(), 3) + 1j * round(rand.normal(), 3)

    return BaseSpinOperator(S, Sx=Sx, Sy=Sy, Sz=Sz, coeff=coeff)


def _random_fermionic_label(k, weights=None):
    """
    Generate a random fermionic label

    Args:
        k: int, length of the label
        weights:

    Returns:

    """
    return ''.join(random.choices(['I', 'N', 'E', '+', '-'], weights, k=k))


def random_fermionic_operator(register_length):
    coeff = round(rand.normal(), 3) + 1j * round(rand.normal(), 3)
    label = _random_fermionic_label(register_length)
    return BaseFermionOperator(label, coeff)
