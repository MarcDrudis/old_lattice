# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from .qiskit_lattice_error import QiskitLatticeError
from .lattice import Lattice
import numpy as np

__all__ = ['single_site_pauli', 'single_edge_pauli', 'site_I', 'site_X',
           'site_Y', 'site_Z', 'site_plus', 'site_minus', 'edge_I', 'edge_X',
           'edge_Y', 'edge_Z', 'edge_plus', 'edge_minus', 'site_create', 'site_destroy']


# --------------------- 1. Pauli Operators on Lattice Sites ------------------------------------------------------------
def single_site_pauli(pauli: str, lattice, site: list, output_format='Operator'):
    """Implements basic pauli gates on a single lattice site."""
    site_index = lattice.site_index(site)  # get the index of the site (int)

    # generate the Pauli string label (identity except for the given `pauli` at site `site_index`
    label = 'I' * (lattice.nedges + lattice.nsites - site_index - 1) + pauli + 'I' * site_index
    # label = label[::-1]
    if output_format == 'Pauli':
        return Pauli(label=label)
    elif output_format == 'Operator':
        return WeightedPauliOperator([[1., Pauli(label=label)]])


def site_I(lattice, site, output_format='Operator'):
    """Implements identity on `site`"""
    return single_site_pauli('I', lattice, site, output_format)


def site_X(lattice, site, output_format='Operator'):
    """Implements Pauli X gate on `site`"""
    return single_site_pauli('X', lattice, site, output_format)


def site_Y(lattice, site, output_format='Operator'):
    """Implements Pauli Y on `site`"""
    return single_site_pauli('Y', lattice, site, output_format)


def site_Z(lattice, site, output_format='Operator'):
    """Implements Pauli Z on `site`"""
    return single_site_pauli('Z', lattice, site, output_format)


def site_plus(lattice, site, output_format='Operator'):
    """Implements sigma + operator on `site`"""
    return 0.5 * (site_X(lattice, site) + 1j * site_Y(lattice, site))


def site_minus(lattice, site, output_format='Operator'):
    """Implements sigma - operator on `site`"""
    return 0.5 * (site_X(lattice, site) - 1j * site_Y(lattice, site))


# --------------------- 2. Pauli Operators on Lattice Edges ------------------------------------------------------------
def single_edge_pauli(pauli: str, lattice, edge: list, output_format='Operator'):
    """Implements basic pauli gates on a single edge."""
    edge_index = lattice.edge_index(edge)
    label = 'I' * (lattice.nedges - edge_index - 1) + pauli + 'I' * (lattice.nsites + edge_index)
    # label = label[::-1]
    if output_format == 'Pauli':
        return Pauli(label=label)
    elif output_format == 'Operator':
        return WeightedPauliOperator([[1., Pauli(label=label)]])


def edge_I(lattice, edge, output_format='Operator'):
    """Implements identity on  `edge`"""
    return single_edge_pauli('I', lattice, edge, output_format)


def edge_X(lattice, edge, output_format='Operator'):
    """Implements Pauli X on `edge`"""
    return single_edge_pauli('X', lattice, edge, output_format)


def edge_Y(lattice, edge, output_format='Operator'):
    """Implements Pauli Y on `edge`"""
    return single_edge_pauli('Y', lattice, edge, output_format)


def edge_Z(lattice, edge, output_format='Operator'):
    """Implements Pauli Z on `edge`"""
    return single_edge_pauli('Z', lattice, edge, output_format)


def edge_plus(lattice, edge, output_format='Operator'):
    """Implements sigma + operator on `edge`"""
    return 0.5 * (edge_X(lattice, edge) + 1j * edge_Y(lattice, edge))


def edge_minus(lattice, edge, output_format='Operator'):
    """Implements sigma - operator on `edge`"""
    return 0.5 * (edge_X(lattice, edge) - 1j * edge_Y(lattice, edge))


# --------------------- 3. Fermionic Operators on Lattice Sites --------------------------------------------------------
def _jordan_wigner_mode(n: int) -> list:
    """
    Jordan Wigner transformation of Pauli strings of length `n` used for mapping fermionic operators on qubits.

    Args:
        n (int): number of modes/sites for the Jordan-Wigner transform

    Returns:
        a (list): A list of `n` tuples, with each tuple containing two Pauli strings.
            For example, a[j] is the tuple of Jordan-Wigner transformed X and Y operators at qubit j.
            Therefore the creation operator at index j would be given by 0.5*(a[j][0] + 1j*a[j][1]),
            while the annihilation operator would be 0.5*(a[j][0] - 1j*a[j][1])
    """
    a = []
    for i in range(n):
        a_z = np.asarray([1] * i + [0] + [0] * (n - i - 1), dtype=bool)
        a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)
        b_z = np.asarray([1] * i + [1] + [0] * (n - i - 1), dtype=bool)
        b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)
        a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
    return a


def _parity_mode(n: int) -> list:
    """
    Parity transformation of Pauli strings of length `n` used for mapping fermionic operators on qubits.

    Args:
        n (int): number of modes

    Returns:
        a (list): A list of `n` tuples, with each tuple containing two Pauli strings.
            For example, a[j] is the tuple of Parity transformed X and Y operators at qubit j.
            Therefore the creation operator at index j would be given by 0.5*(a[j][0] + 1j*a[j][1]),
            while the annihilation operator would be 0.5*(a[j][0] - 1j*a[j][1])
    """
    a = []
    for i in range(n):
        a_z = [0] * (i - 1) + [1] if i > 0 else []
        a_x = [0] * (i - 1) + [0] if i > 0 else []
        b_z = [0] * (i - 1) + [0] if i > 0 else []
        b_x = [0] * (i - 1) + [0] if i > 0 else []
        a_z = np.asarray(a_z + [0] + [0] * (n - i - 1), dtype=bool)
        a_x = np.asarray(a_x + [1] + [1] * (n - i - 1), dtype=bool)
        b_z = np.asarray(b_z + [1] + [0] * (n - i - 1), dtype=bool)
        b_x = np.asarray(b_x + [1] + [1] * (n - i - 1), dtype=bool)
        a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))

    return a


def site_create(lattice, site, map_type='jordan_wigner'):
    """
    Returns qiskit operator corresponding to the fermionic creation operator at the given `site` of the `lattice`.

    Args:
        lattice (Lattice): The lattice on which the creation operator lives, a qiskit Lattice object
        site (list): The site for which the operator should correspond to a fermionic creation operator
        map_type (str): The fermion-to-qubit mapping used in the transform. Must be one of ['jordan_wigner',
            'parity']

    Returns:
        (WeightedPauliOperator): A qiskit.aqua.operators WeightedPauliOperator representing the fermionic
            creation operator at the given `site` of `lattice`.
    """

    # Generate the transformed pauli stings for all lattice sites with the given mapping
    if map_type == 'jordan_wigner':  # TODO: Can be made more efficient by computing a outside this func for lattice
        a = _jordan_wigner_mode(lattice.nsites + lattice.nedges)
    elif map_type == 'parity':
        a = _parity_mode(lattice.nsites + lattice.nedges)
    else:
        raise QiskitLatticeError('Please specify the supported modes: '
                                 'jordan_wigner, parity, bravyi_kitaev, bksf')

    # Get the index of the specified lattice site from the lattice
    site_index = lattice.site_index(site)

    # Generate the final Pauli string:
    real_part = [0.5, a[site_index][0]]
    imag_part = [0.5j, a[site_index][1]]
    pauli_str = [real_part, imag_part]

    return WeightedPauliOperator(paulis=pauli_str)


def site_destroy(lattice, site, map_type='jordan_wigner'):
    """
    Returns qiskit operator corresponding to the fermionic annihilation operator at the given `site` of the `lattice`.

    Args:
        lattice (Lattice): The lattice on which the annihilation operator lives, a qiskit Lattice object
        site (list): The site for which the operator should correspond to a fermionic annihilation operator
        map_type (str): The fermion-to-qubit mapping used in the transform. Must be one of ['jordan_wigner',
            'parity']

    Returns:
        (WeightedPauliOperator): A qiskit.aqua.operators WeightedPauliOperator representing the fermionic
            annihilation operator at the given `site` of `lattice`.
    """

    # Generate the transformed pauli stings for all lattice sites with the given mapping
    if map_type == 'jordan_wigner':  # TODO: Can be made more efficient by computing a outside this func for lattice
        a = _jordan_wigner_mode(lattice.nsites + lattice.nedges)
    elif map_type == 'parity':
        a = _parity_mode(lattice.nsites + lattice.nedges)
    else:
        raise QiskitLatticeError('Please specify the supported modes: '
                                 'jordan_wigner, parity, bravyi_kitaev, bksf')

    # Get the index of the specified lattice site from the lattice
    site_index = lattice.site_index(site)

    # Generate the final Pauli string:
    real_part = [0.5, a[site_index][0]]
    imag_part = [-0.5j, a[site_index][1]]
    pauli_str = [real_part, imag_part]

    return WeightedPauliOperator(paulis=pauli_str)
