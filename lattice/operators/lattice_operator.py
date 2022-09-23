# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import numpy as np

from ..qiskit_lattice_error import QiskitLatticeError
from ..lattice import Lattice
from .fermionic_operators import ParticleTypeOperator

logger = logging.getLogger(__name__)

fermion_mappings = ('jordan_wigner', 'parity')


# operator_types =  [FermionicOperator, SpinOperator]


class LatticeOperator:
    """
    A class to bundle together different types of operators that can exist on a lattice (e.g. FermionOperators,
    SpinOperators, ...) and to provide a mapping method that maps all of these operators together to QubitOperators
    (qiskit.aqua Operators) that can be used on QuantumHardware.
    """

    def __init__(self, lattice, operator_list=[], fermion_map_type='jordan_wigner'):
        """Constructor.

        Args:
            operator_list (list): A list of operators that constitue the given `LatticeOperator`.
        """
        assert isinstance(lattice, Lattice), '`lattice` must be one a valid qiskit Lattice object'
        self.lattice = lattice

        assert isinstance(operator_list, list), '`operator_list` must be a list of operators'
        self._operator_list = operator_list

        assert fermion_map_type in fermion_mappings, '`fermion_map_type` must be one of {0}'.format(fermion_mappings)
        self._fermion_map_type = fermion_map_type

    @property
    def nsites(self):
        """Getter of number of lattice sites."""
        return self.lattice.nsites

    @property
    def nedges(self):
        """Getter of number of lattice edges"""
        return self.lattice.nedges

    @property
    def operator_list(self):
        """Getter of operator list"""
        return self._operator_list

    @operator_list.setter
    def operator_list(self, new_operator_list):
        """Setter of operator list."""
        assert isinstance(new_operator_list, list), '`operator_list` must be a list of operators'
        self._operator_list = new_operator_list

    @property
    def fermion_map_type(self):
        """Getter of one fermion map type."""
        return self._fermion_map_type

    @fermion_map_type.setter
    def fermion_map_type(self, new_map_type):
        """Setter of fermion map type."""
        assert new_map_type in fermion_mappings, '`fermion_map_type` must be one of {0}'.format(fermion_mappings)
        self._fermion_map_type = new_map_type

    def __add__(self, other):
        """
        Add two `LatticeOperator` instances by appending their operator_lists.

        Args:
            other (LatticeOperator): The `LatticeOperator` object to add.

        Returns:
            LatticeOperator
        """
        if isinstance(other, LatticeOperator):
            assert self.lattice == other.lattice, 'Cannot add Operators that operator on different lattices.'
            assert self.fermion_map_type == other.fermion_map_type, 'Cannot add Operators with incompatible mappings'
            combined_operator_list = self.operator_list + other.operator_list
        elif isinstance(other, ParticleTypeOperator):
            self._operator_list.append(other)
            combined_operator_list = self.operator_list + [other]
        else:
            raise QiskitLatticeError('{0} is not a valid `LatticeOperator` object`'.format(other))

        # Return a Lattice operator with a combined operator list
        return LatticeOperator(lattice=self.lattice,
                               operator_list=combined_operator_list,
                               fermion_map_type=self.fermion_map_type)

    def __iadd__(self, other):
        """
        Add the `other` LatticeOperator instance to `self` by appending the operator list.

        Args:
            other (LatticeOperator): The `LatticeOperator` object to add.

        Returns:
            LatticeOperator
        """
        if isinstance(other, LatticeOperator):
            assert self.lattice == other.lattice, 'Cannot add Operators that operator on different lattices.'
            assert self.fermion_map_type == other.fermion_map_type, 'Cannot add Operators with incompatible mappings'
            self._operator_list.append(other.operator_list)
        elif isinstance(other, ParticleTypeOperator):
            self._operator_list.append(other)
        elif isinstance(other, list):
            for elem in other:
                self.__iadd__(other)
        else:
            raise QiskitLatticeError('{0} is not a valid `LatticeOperator` object`'.format(other))

        # Add the list of operators

        return self
