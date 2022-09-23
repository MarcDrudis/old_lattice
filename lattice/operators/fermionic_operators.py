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

import numpy as np
import qutip
from .qiskit_aqua_operator_utils import *
from .particle_type_operator import ParticleTypeOperator
# TODO: Nicer handling of zero operator.
default_mode = 'jordan_wigner'

class FermionicOperator(ParticleTypeOperator):
    """
    Fermionic type operators

    The abstract fermionic registers are implemented in two subclasses,BaseFermionOperatorandFermionicOperator,
    inspired by the implementation of Pauli operators onqiskit. ABaseFermionOperatoris the equivalent of a single Pauli
    string on a qubit register. TheFermionicOperatorrepresentsa sum of multipleBaseFermionOperators. They act on
    fermionic registers of a fixed length deter-mined at the time of initialization.
    """

    def __init__(self, operator_list, register_length=None):

        # 0. Initialize member variables
        if not any(True for _ in operator_list):
            # Treat case of zero operator (empty operator_list)
            assert isinstance(register_length, int), 'When instantiating the zero FermionicOperator, a register' \
                                                     'length must be provided.'
            self._register_length = register_length
        else:
            # Treat case of nonzero operator_list
            self._register_length = copy.deepcopy(len(operator_list[0]))

        self._operator_dict = {}

        # Go through all operators in the operator list
        for base_operator in operator_list:
            # 1.  Parse
            # 1.1 Check if they are valid, compatible BaseFermionOperator instances
            assert isinstance(base_operator, BaseFermionOperator), 'FermionicOperators must be built up from ' \
                                                          '`BaseFermionOperator` objects'
            assert len(base_operator) == self._register_length, 'FermionicOperators must act on fermionic registers ' \
                                                         'of same length.'

            # 2.  Add the valid operator to self._operator_dict
            # 2.2 If the operator has zero coefficient, skip the rest of the steps
            if base_operator.coeff == 0:
                continue

            # 2.3 For nonzero coefficient, add the operator the the dictionary of operators
            operator_label = base_operator.label
            if operator_label not in self._operator_dict.keys():
                # If an operator of the same signature (label) is not yet present in self._operator_dict, add it.
                self._operator_dict[operator_label] = base_operator
            else:
                # Else if an operator of the same signature exists already, add the coefficients.
                self._operator_dict[operator_label].coeff += base_operator.coeff

                # If after addition the coefficient is 0, remove the operator from the list
                if self._operator_dict[operator_label].coeff == 0:
                    self._operator_dict.pop(operator_label)

        # 3. Set the particle type
        ParticleTypeOperator.__init__(self, particle_type='fermionic')

    def __repr__(self):
        """Sets the representation of `self` in the console."""

        # 1. Treat the case of the zero-operator:
        if self.operator_list == []:
            return 'zero operator ({})'.format(self.register_length)

        # 2. Treat the general case:
        full_str = ''
        for operator in self.operator_list:
            full_str += '{1} \t {0}\n'.format(operator.coeff, operator.label)
        return full_str

    def __mul__(self, other):
        """
        Overloads the multiplication operator `*` for self and other, where other is a number-type,
        a BaseFermionOperator or a FermionicOperator.
        """
        # Catch the case of a zero FermionicOperator (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BaseFermionOperator):
                assert self._register_length == len(other), "Operators act on Fermion Registers of different length"
            elif isinstance(other, FermionicOperator):
                assert self._register_length == other._register_length, "Operators act on Fermion Registers" \
                                                                        " of different length"
            # return BaseFermionOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, (numbers.Number, BaseFermionOperator)):
            # Create copy of the FermionicOperator in which every BaseFermionOperator is multiplied by `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other for base_operator in self.operator_list]
            return FermionicOperator(new_operatorlist)

        elif isinstance(other, FermionicOperator):
            # Initialize new operator_list for the returned Fermionic operator
            new_operatorlist = []

            # Catch the case of a zero FermionicOperator (for `other`)
            if not any(True for _ in other._operator_dict):
                assert self._register_length == other._register_length, "Operators act on Fermion Registers " \
                                                                        "of different length"
                return other

            # Compute the product (Fermionic type operators consist of a sum of BaseFermionOperator)
            # F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci are BaseFermion type Operators
            for op1 in self.operator_list:
                for op2 in other.operator_list:
                    new_operatorlist.append(op1 * op2)
            return FermionicOperator(new_operatorlist)

        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects or
        BaseFermionOperators
        """
        # Catch the case of a zero FermionicOperator (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BaseFermionOperator):
                assert self._register_length == len(other), "Operators act on Fermion Registers of different length"
            # return BaseFermionOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        elif isinstance(other, BaseFermionOperator):
            # Create copy of the FermionicOperator in which `other` is multiplied by every BaseFermionOperator
            new_operatorlist = [other * copy.deepcopy(base_operator) for base_operator in self.operator_list]
            return FermionicOperator(new_operatorlist)

        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'FermionicOperator'".format(type(other).__name__))

    def __truediv__(self, other):
        """
        Overloads the division operator `/` for division by number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)
        else:
            raise TypeError(
                "unsupported operand type(s) for /: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def __add__(self, other):
        """Returns a `FermionicOperator` representing the sum of the given base fermionic operators."""

        if isinstance(other, BaseFermionOperator):
            # Create copy of the FermionicOperator
            new_operatorlist = copy.deepcopy(self.operator_list)

            # Only add the new operator if it has a nonzero-coefficient.
            if not other.coeff == 0:
                # Check compatiblility (i.e. operators act on same register length)
                assert self._is_compatible(other), "Incompatible register lengths for '+'. "
                new_operatorlist.append(other)

            return FermionicOperator(new_operatorlist)

        elif isinstance(other, FermionicOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            other_operatorlist = copy.deepcopy(other.operator_list)

            # Check compatiblility (i.e. operators act on same register length)
            assert self._is_compatible(other), "Incompatible register lengths for '+'. "

            new_operatorlist += other_operatorlist

            return FermionicOperator(new_operatorlist)

        else:
            raise TypeError(
                "unsupported operand type(s) for +: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a `FermionicOperator` representing the difference of the given fermionic operators."""
        if isinstance(other, (BaseFermionOperator, FermionicOperator)):
            return self.__add__(-1 * other)

        else:
            raise TypeError(
                "unsupported operand type(s) for -: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def __pow__(self, power):
        """
        Overloads the power operator `**` for applying an operator `self` `power` number of times, e.g. op^{power}
        where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")
            elif power == 0:
                identity = FermionicOperator([BaseFermionOperator('I' * self._register_length)])
                return identity
            else:
                operator = copy.deepcopy(self)
                for k in range(power-1):
                    operator *= operator
                return operator
        else:
            raise TypeError(
                "unsupported operand type(s) for **: 'FermionicOperator' and '{}'".format(type(power).__name__))

    @property
    def operator_list(self):
        """Getter for the operator_list of `self`"""
        return list(self._operator_dict.values())

    @property
    def register_length(self):
        """Getter for the length of the fermionic register that the FermionicOperator `self` acts on."""
        return self._register_length

    def dag(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        daggered_operator_list = [operator.dag() for operator in self.operator_list]
        return FermionicOperator(daggered_operator_list)

    def _is_compatible(self, operator):
        """
        Checks whether the `operator` is compatible (same shape and

        Args:
            operator (BaseFermionOperator/FermionicOperator):

        Returns:
            bool,
                True iff `operator` is compatible with `self`.
        """
        same_length = (self.register_length == operator.register_length)
        compatible_type = isinstance(operator, (BaseFermionOperator, FermionicOperator))

        if not compatible_type or not same_length:
            return False

        return True

    def to_qiskit(self, mode=default_mode):
        """
        Returns the qubit transformation of `self` as a qiskit.aqua.operators WeightedPauliOperator object.
        """
        if mode == 'jordan_wigner':
            transformed_XY = BaseFermionOperator('I'*self.register_length)._jordan_wigner_mode()
        elif mode == 'bravyi_kitaev':
            transformed_XY = BaseFermionOperator('I'*self.register_length)._bravyi_kitaev_mode()
        elif mode == 'parity':
            transformed_XY = BaseFermionOperator('I' * self.register_length)._bravyi_kitaev_mode()
        else:
            raise UserWarning("`mode` must be one of ['jordan_wigner', 'bravyi_kitaev', 'parity]")

        aqua_operator = operator_sum([operator.to_qiskit(custom_transformed_XY=transformed_XY)
                                      for operator in self.operator_list])

        # Chop potential zero entries
        aqua_operator.chop()

        return aqua_operator

    def to_qutip(self, mode=default_mode, force=False):
        """
        Returns the qutip.Quobj which represents the given FermionicOperator  `self`.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the qutip.Qobj is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            qutip.Qobj
        """
        return operator_sum([operator.to_qutip(mode, force) for operator in self.operator_list])

    def to_matrix(self, mode=default_mode, force=False):
        """Returns a dense numpy matrix representing `self` in matrix form.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            np.ndarray
        """
        return sum([operator.to_matrix(mode, force) for operator in self.operator_list])

    def to_spmatrix(self, mode=default_mode, force=False):
        """Returns a sparse numpy matrix representing `self` in matrix form.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            numpy sparse matrix
        """
        return sum([operator.to_spmatrix(mode, force) for operator in self.operator_list])

    def to_qubit_operator(self, output='qiskit', **kwargs):
        """
        Wrapper function for the conversion of a `FermionicOperator` to an operator that acts on qubits.
        The output type depends on the `output` argument as described below.

        Args:
            output: string,
                The desired output type. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']
            **kwargs:
                The keyword arguments to pass to the respective mapping belonging to the `output`

        Returns:
            various types
            The following types are returned depending on the `output` variable:
                'qiskit'   - qiskit.aqua.Operator
                'qutip'    - qutip.Qobj
                'matrix'   - np.ndarray
                'spmatrix' - numpy sparse matrix
        """
        if output == 'qiskit':
            return self.to_qiskit(**kwargs)
        elif output == 'qutip':
            return self.to_qutip(**kwargs)
        elif output == 'matrix':
            return self.to_matrix(**kwargs)
        elif output == 'spmatrix':
            return self.to_spmatrix(**kwargs)
        else:
            supported_output = ['qiskit', 'qutip', 'matrix', 'spmatrix']
            raise TypeError("output type '{}' is not supported. Please use one of {}".format(output, supported_output))


class BaseFermionOperator(ParticleTypeOperator):
    """
    A class for simple products (not sum) of fermionic operators on several fermionic modes.
    """

    def __init__(self, label, coeff=1.):

        # 1. Parse input
        # Parse coeff
        if not isinstance(coeff, numbers.Number):
            raise TypeError("`coeff` must be a number type not '{}'".format(type(coeff).__name__))

        # Parse label
        for char in label:
            # I encodes the identity
            # + the creation operator in the fermion mode at the given position
            # - the annihilation operator in the fermion mode at the given position
            # N denotes the ocupation number operator
            # E denotes 1-N (the `emptiness number` operator)
            assert char in ['I', '+', '-', 'N',
                            'E'], "Label must be a string consisting only of ['I','+','-','N','E'] not: {}".format(char)

        # 2. Initialize member variables
        self.coeff = coeff
        self.transformed_XY = None
        self.label = label

        # 3. Set the particle type
        ParticleTypeOperator.__init__(self, particle_type='fermionic')

    def __len__(self) -> int:
        """
        Returns the number of of fermion modes in the fermionic register, i.e. the length of `self.label`
        """
        return len(self.label)

    def __repr__(self) -> str:
        # 1. Treat the case of the zero operator
        if self.coeff == 0:
            return 'zero operator ({})'.format(len(self.label))

        # 2. Treat the general case
        return '{1} \t {0}'.format(self.coeff, self.label)

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects
        """
        if isinstance(other, numbers.Number):
            return BaseFermionOperator(label=self.label, coeff=other * self.coeff)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'BaseFermionOperator'".format(type(other).__name__))

    def __mul__(self, other):
        """
        Overloads the multiplication operator `*` for self and other, where other is a number-type or
        a BaseFermionicOperator
        """
        if isinstance(other, numbers.Number):
            return BaseFermionOperator(label=self.label, coeff=other * self.coeff)

        elif isinstance(other, BaseFermionOperator):
            assert len(self) == len(other), "Operators act on Fermion Registers of different length"

            new_coeff = self.coeff * other.coeff
            new_label = ''

            # Map the products of two operators on a single fermionic mode to their result.
            mapping = {
                # 0                   - if product vanishes,
                # `newlabel`           - if product does not vanish
                'II': 'I',
                'I+': '+',
                'I-': '-',
                'IN': 'N',
                'IE': 'E',

                '+I': '+',
                '++': 0,
                '+-': 'N',
                '+N': 0,
                '+E': '+',

                '-I': '-',
                '-+': 'E',
                '--': 0,
                '-N': '-',
                '-E': 0,

                'NI': 'N',
                'N+': '+',
                'N-': 0,
                'NN': 'N',
                'NE': 0,

                'EI': 'E',
                'E+': 0,
                'E-': '-',
                'EN': 0,
                'EE': 'E'
            }

            for i, char1, char2 in zip(np.arange(len(self)), self.label, other.label):

                # if char2 is one of `-`, `+` we pick up a phase when commuting it to the position of char1
                if char2 in ['-', '+']:
                    # Construct the string through which we have to commute
                    permuting_through = self.label[i + 1:]
                    # Count the number of times we pick up a minus sign when commuting
                    ncommutations = permuting_through.count('+') + permuting_through.count('-')
                    new_coeff *= (-1) ** ncommutations

                # Check what happens to the symbol
                new_char = mapping[char1 + char2]
                if new_char is 0:
                    return BaseFermionOperator('I'*len(self), coeff = 0.)
                else:
                    new_label += new_char

            return BaseFermionOperator(new_label, new_coeff)

        # Multiplication with a FermionicOperator is implemented in the __rmul__ method of the FermionicOperator class
        elif isinstance(other, FermionicOperator):
            return NotImplemented

        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'BaseFermionOperator' and '{}'".format(type(other).__name__))

    def __pow__(self, power):
        """
        Overloads the power operator `**` for applying an operator `self` `power` number of times, e.g. op^{power}
        where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")
            elif power == 0:
                identity = BaseFermionOperator('I' * len(self))
                return identity
            else:
                operator = copy.deepcopy(self)
                for k in range(power-1):
                    operator *= operator
                return operator
        else:
            raise TypeError(
                "unsupported operand type(s) for **: 'FermionicOperator' and '{}'".format(type(power).__name__))

    def __add__(self, other) -> FermionicOperator:
        """Returns a fermionic operator representing the sum of the given BaseFermionicOperators"""

        if isinstance(other, BaseFermionOperator):
            # Case 1: `other` is a `BaseFermionOperator`.
            #  In this case we add the two operators, if they have non-zero coefficients. Otherwise we simply
            #  return the operator that has a non-vanishing coefficient (self, if both vanish).
            if other.coeff == 0:
                return copy.deepcopy(self)
            elif self.coeff == 0:
                return copy.deepcopy(other)
            elif self.label == other.label:
                return BaseFermionOperator(self.label, self.coeff + other.coeff)
            else:
                return FermionicOperator([copy.deepcopy(self), copy.deepcopy(other)])
        elif isinstance(other, FermionicOperator):
            # Case 2: `other` is a `FermionicOperator`.
            #  In this case use the __add__ method of FermionicOperator.
            return other.__add__(self)
        else:
            # Case 3: `other` is any other type. In this case we raise an error.
            raise TypeError(
                "unsupported operand type(s) for +: 'BaseFermionOperator' and '{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a fermionic operator representing the sum of the given BaseFermionicOperators"""

        if not isinstance(other, BaseFermionOperator):
            raise TypeError(
                "unsupported operand type(s) for -: 'BaseFermionOperator' and '{}'".format(type(other).__name__))

        return self.__add__(-1 * other)

    def __truediv__(self, other):
        """
        Overloads the division operator `/` for division by number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)
        else:
            raise TypeError(
                "unsupported operand type(s) for /: 'FermionicOperator' and '{}'".format(type(other).__name__))

    @property
    def register_length(self):
        return len(self.label)

    def is_normal(self) -> bool:
        """
        Returns True iff `self.label` is normal ordered.

        Returns:
            bool, True iff the product `self.label` is normal ordered (i.e. - on each mode to the right, + to left)
        """
        return self.label.count('E') == 0

    def normal_order(self) -> list:
        """
        Returns the list of normal-order components of `self`.
        """
        # Catch the case of the zero-operator:
        if self.coeff == 0:
            return []

        # Set up an empty list in which to save the normal ordered expansion
        normal_ordered_operator_list = []

        # Split the `self.label` at every non-normal ordered symbol (only E = -+)
        splits = self.label.split('E')
        # Count the number of splits
        nsplits = self.label.count('E')

        # Generate all combinations of (I,N) of length nsplits
        combos = list(map(''.join, itertools.product('IN', repeat=nsplits)))

        for combo in combos:
            # compute the sign of the given label combination
            sign = (-1) ** combo.count('N')

            # build up the label token
            label = splits[0]
            for link, next_base in zip(combo, splits[1:]):
                label += link + next_base
            # append the current normal ordered part to the list of the normal ordered expansion
            normal_ordered_operator_list.append(BaseFermionOperator(label=label, coeff=sign * self.coeff))

        return normal_ordered_operator_list

    def dag(self):
        """Returns the adjoint (dagger) of `self`."""
        daggered_label = ''

        dagger_map = {
            '+': '-',
            '-': '+',
            'I': 'I',
            'N': 'N',
            'E': 'E'
        }

        phase = 1.
        for i, char in enumerate(self.label):
            daggered_label += dagger_map[char]
            if char in ['+', '-']:
                permute_through = self.label[i+1:]
                phase *= (-1)**( permute_through.count('+') + permute_through.count('-') )

        return BaseFermionOperator(label=daggered_label, coeff=phase * np.conj(self.coeff))

    def _jordan_wigner_mode(self) -> list:
        """
        Generates a lookup table for the Jordan Wigner transformation of FermionOperator strings of length
        `len(self)` used for mapping fermionic operators on qubits.

        Returns:
            lookup_table_XY (list): A list with length  `len(self)` (= number of modes) 2-tuples, with each
                2-tuple containing two Pauli strings:
                For example, lookup_table_XY[j] is the 2-tuple of Jordan-Wigner transformed X and Y
                operators at qubit j.

                Therefore the creation operator at index j would be given by
                    0.5*(lookup_table_XY[j][0] + 1j*lookup_table_XY[j][1]),
                while the annihilation operator would be
                    0.5*(lookup_table_XY[j][0] - 1j*lookup_table_XY[j][1])
        """

        # Check if the lookup table has not already been initialized.
        if self.transformed_XY is None:
            nmodes = len(self)  # number of modes/sites for the Jordan-Wigner transform (= number of fermionc modes)
            lookup_table_XY = []  # initialize empty lookup table
            for i in range(nmodes):
                a_z = np.asarray([1] * i + [0] + [0] * (nmodes - i - 1), dtype=bool)
                a_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                b_z = np.asarray([1] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                b_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                c_z = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                c_x = np.asarray([0] * nmodes, dtype=bool)
                lookup_table_XY.append((Pauli(a_z, a_x),
                                        Pauli(b_z, b_x)))
                # add Pauli 3-tuple to lookup table

            # Save the lookup table in `self.transformed_XY`
            self.transformed_XY = lookup_table_XY

        return self.transformed_XY

    def _parity_mode(self) -> list:
        """
         Generates a lookup table for the parity transformation of FermionOperator strings of length
        `len(self)` used for mapping fermionic operators on qubits.

        Returns:
            lookup_table_XY (list): A list with length  `len(self)` (= number of modes) 2-tuples, with each
                2-tuple containing two Pauli strings:
                For example, lookup_table_XY[j] is the 2-tuple of parity transformed X and Y
                operators at qubit j.

                Therefore the creation operator at index j would be given by
                    0.5*(lookup_table_XY[j][0] + 1j*lookup_table_XY[j][1]),
                while the annihilation operator would be
                    0.5*(lookup_table_XY[j][0] - 1j*lookup_table_XY[j][1])
        """
        # Check if the lookup table has not already been initialized.
        if self.transformed_XY is None:
            nmodes = len(self)  # number of modes/sites for the parity transform (= number of fermionic modes)
            lookup_table_XY = []  # initialize empty lookup table

            for i in range(nmodes):
                a_z = [0] * (i - 1) + [1] if i > 0 else []
                a_x = [0] * (i - 1) + [0] if i > 0 else []
                b_z = [0] * (i - 1) + [0] if i > 0 else []
                b_x = [0] * (i - 1) + [0] if i > 0 else []
                a_z = np.asarray(a_z + [0] + [0] * (nmodes - i - 1), dtype=bool)
                a_x = np.asarray(a_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
                b_z = np.asarray(b_z + [1] + [0] * (nmodes - i - 1), dtype=bool)
                b_x = np.asarray(b_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
                lookup_table_XY.append((Pauli(a_z, a_x),
                                        Pauli(b_z, b_x)))

            # Save the lookup table in `self.transformed_XY`
            self.transformed_XY = lookup_table_XY

        return self.transformed_XY

    def _bravyi_kitaev_mode(self) -> list:
        """
        Generates a lookup table for the Bravyi Kitaev transformation of FermionOperator strings of length
        `len(self)` used for mapping fermionic operators on qubits.

        Returns:
            lookup_table_XY (list): A list with length  `len(self)` (= number of modes) 2-tuples, with each
                2-tuple containing two Pauli strings:
                For example, lookup_table_XY[j] is the 2-tuple of Bravyi Kitaev transformed X and Y
                operators at qubit j.

                Therefore the creation operator at index j would be given by
                    0.5*(lookup_table_XY[j][0] + 1j*lookup_table_XY[j][1]),
                while the annihilation operator would be
                    0.5*(lookup_table_XY[j][0] - 1j*lookup_table_XY[j][1])
        """

        # Check if the lookup table has not already been initialized.
        if self.transformed_XY is None:
            nmodes = len(self)  # number of modes/sites for the Bravyi Kitaev transform (= number of fermionc modes)
            lookup_table_XY = []  # initialize empty lookup table

            def parity_set(j, n):
                """Computes the parity set of the j-th orbital in n modes.
                Args:
                    j (int) : the orbital index
                    n (int) : the total number of modes
                Returns:
                    numpy.ndarray: Array of mode indexes
                """
                indexes = np.array([])
                if n % 2 != 0:
                    return indexes

                if j < n / 2:
                    indexes = np.append(indexes, parity_set(j, n / 2))
                else:
                    indexes = np.append(indexes, np.append(
                        parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
                return indexes

            def update_set(j, n):
                """Computes the update set of the j-th orbital in n modes.
                Args:
                    j (int) : the orbital index
                    n (int) : the total number of modes
                Returns:
                    numpy.ndarray: Array of mode indexes
                """
                indexes = np.array([])
                if n % 2 != 0:
                    return indexes
                if j < n / 2:
                    indexes = np.append(indexes, np.append(
                        n - 1, update_set(j, n / 2)))
                else:
                    indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
                return indexes

            def flip_set(j, n):
                """Computes the flip set of the j-th orbital in n modes.
                Args:
                    j (int) : the orbital index
                    n (int) : the total number of modes
                Returns:
                    numpy.ndarray: Array of mode indexes
                """
                indexes = np.array([])
                if n % 2 != 0:
                    return indexes
                if j < n / 2:
                    indexes = np.append(indexes, flip_set(j, n / 2))
                elif j >= n / 2 and j < n - 1:  # pylint: disable=chained-comparison
                    indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
                else:
                    indexes = np.append(np.append(indexes, flip_set(
                        j - n / 2, n / 2) + n / 2), n / 2 - 1)
                return indexes

            # FIND BINARY SUPERSET SIZE
            bin_sup = 1
            while nmodes > np.power(2, bin_sup):
                bin_sup += 1
            # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
            update_sets = []
            update_pauli = []

            parity_sets = []
            parity_pauli = []

            flip_sets = []

            remainder_sets = []
            remainder_pauli = []
            for j in range(nmodes):

                update_sets.append(update_set(j, np.power(2, bin_sup)))
                update_sets[j] = update_sets[j][update_sets[j] < nmodes]

                parity_sets.append(parity_set(j, np.power(2, bin_sup)))
                parity_sets[j] = parity_sets[j][parity_sets[j] < nmodes]

                flip_sets.append(flip_set(j, np.power(2, bin_sup)))
                flip_sets[j] = flip_sets[j][flip_sets[j] < nmodes]

                remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

                update_pauli.append(Pauli(np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool)))
                parity_pauli.append(Pauli(np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool)))
                remainder_pauli.append(Pauli(np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool)))
                for k in range(nmodes):
                    if np.in1d(k, update_sets[j]):
                        update_pauli[j].update_x(True, k)
                    if np.in1d(k, parity_sets[j]):
                        parity_pauli[j].update_z(True, k)
                    if np.in1d(k, remainder_sets[j]):
                        remainder_pauli[j].update_z(True, k)

                x_j = Pauli(np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool))
                x_j.update_x(True, j)
                y_j = Pauli(np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool))
                y_j.update_z(True, j)
                y_j.update_x(True, j)

                x_pauli = update_pauli[j] * x_j * parity_pauli[j]
                y_pauli = update_pauli[j] * y_j * remainder_pauli[j]

                lookup_table_XY.append((x_pauli,
                                         y_pauli,
                                         x_pauli))

            # Save the lookup table in `self.transformed_XY`
            self.transformed_XY = lookup_table_XY

        return self.transformed_XY

    def to_qiskit(self, mode=default_mode, custom_transformed_XY = None) -> WeightedPauliOperator:
        """
        Returns the  transformation of `self` as a qiskit.aqua.operators WeightedPauliOperator object.
        """

        # Catch the exception of a zero operator:
        if len(self) == 0 or self.coeff == 0:
            return WeightedPauliOperator(paulis = [])

        # 0. Set up the lookup table `self.transformed_XY` for the JW transform
        if custom_transformed_XY is not None:
            self.transformed_XY = custom_transformed_XY

        if self.transformed_XY is None:
            if mode == 'jordan_wigner':
                self._jordan_wigner_mode()
            elif mode == 'bravyi_kitaev':
                self._bravyi_kitaev_mode()
            elif mode == 'parity':
                self._parity_mode()
            else:
                raise UserWarning("mode must be one of ['jordan_wigner', 'bravyi_kitaev', 'parity']")

        # 1. Initialize an operator list with the identity scaled by the `self.coeff`
        all_false = np.asarray([False] * len(self), dtype=bool)

        scaled_identity = WeightedPauliOperator(paulis=[[self.coeff,Pauli(z=all_false, x=all_false)]]) # self.coeff * I
        operator_list = [scaled_identity]

        # Go through the label and replace the fermion operators by their qubit-equivalent, then save the
        # respective Pauli string in the pauli_str list.
        for position, char in enumerate(self.label):
            # The creation operator is given by 0.5*(X + 1j*Y)
            if char == '+':
                real_part = [0.5, self.transformed_XY[position][0]]    # 0.5 * X
                imag_part = [0.5j, self.transformed_XY[position][1]]   # 0.5j * Y
                pauli_str = [real_part, imag_part]
                operator_list.append(WeightedPauliOperator(paulis=pauli_str))

            # The annihilation operator is given by 0.5*(X - 1j*Y)
            elif char == '-':
                real_part = [0.5, self.transformed_XY[position][0]]    # 0.5 * X
                imag_part = [-0.5j, self.transformed_XY[position][1]]  # -0.5j * Y
                pauli_str = [real_part, imag_part]
                operator_list.append(WeightedPauliOperator(paulis=pauli_str))

            # The occupation number operator N is given by 0.5*(I + Z)
            elif char == 'N':
                offset_part = [0.5, Pauli(z=all_false, x=all_false)]    # 0.5 * I
                z_part      = [0.5, self.transformed_XY[position][1] * self.transformed_XY[position][0]]  # 0.5 * YX
                pauli_str = [offset_part, z_part]
                operator_list.append(WeightedPauliOperator(paulis=pauli_str))

            # The `emptiness number` operator I - N is given by 0.5*(I - Z)
            elif char == 'E':
                offset_part = [0.5, Pauli(z=all_false, x=all_false)]     # 0.5 * I
                z_part      = [-0.5, self.transformed_XY[position][1] * self.transformed_XY[position][0]]  # -0.5 * Z
                pauli_str = [offset_part, z_part]
                operator_list.append(WeightedPauliOperator(paulis=pauli_str))

            elif char == 'I':
                continue

            # catch any disallowed labels
            else:
                raise UserWarning(
                    "BaseFermionOperator label included '{}'. Allowed characters: I, N, E, +, -".format(char))

        # 2. Multiply all transformed qubit operators to get the final result
        aqua_operator = operator_product(operator_list)
        # Remove pauli strings with 0 coefficient
        aqua_operator.chop()
        return aqua_operator

    @staticmethod
    def _qutip_jordan_wigner_transform(j: int, length: int):
        """Performs the jordan wigner transform of the lowering operator psi_j in arbitrary dimensions in qutip"""

        sigmaz = qutip.sigmaz()
        sigmam = qutip.sigmam()

        identity = qutip.identity(2)
        assert j < length, 'No site %d in lattice of length %d. Indexing from 0.' % (j, length)

        operators = []
        for k in range(j):
            operators.append(sigmaz)
        operators.append(sigmam)
        for k in range(length - j - 1):
            operators.append(identity)

        return qutip.tensor(
            operators[::-1])  # Note: I took a minus sign away here! (maybe bc. of the order of the operators?)
        # This was needed to make it agree with my qiskit transform.
        # Note also that the [::-1] is introduced here because the qiskit and qutip orderings of the tensor
        # product are reversed w.r.t. each other.

    def to_qutip(self, mode=default_mode, force=False):
        """
        Returns the qutip.Quobj which represents the given BaseFermionOperator `self`.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the qutip.Qobj is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            qutip.Qobj
        """
        expected_dim = 1 << len(self)
        if expected_dim > 1e5 and not force:
            raise UserWarning("Expected matrix dimension is {}. Calculation may take signficiant time. "
                              "To proceed set the argument `force` to True.".format(expected_dim))

        if mode == 'jordan_wigner':
            operator = self.coeff * qutip.identity([2] * len(self))

            for i, label in enumerate(self.label):
                if label == 'I':
                    continue
                elif label == '+':
                    operator *= self._qutip_jordan_wigner_transform(i, len(self)).dag()
                elif label == '-':
                    operator *= self._qutip_jordan_wigner_transform(i, len(self))
                elif label == 'N':
                    operator *= self._qutip_jordan_wigner_transform(i, len(self)).dag() \
                                * self._qutip_jordan_wigner_transform(i, len(self))
                elif label == 'E':
                    operator *= self._qutip_jordan_wigner_transform(i, len(self)) \
                                * self._qutip_jordan_wigner_transform(i, len(self)).dag()
        else:
            raise UserWarning("mode must be one of ['jordan_wigner']")
            # TODO: Implement 'bravyi-kitaev' and 'parity' mode for qutip?
        return operator

    def to_matrix(self, mode=default_mode, force=False):
        """Returns a dense numpy matrix representing `self` in matrix form.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            np.ndarray
        """
        return self.to_qutip(mode, force).data.todense()

    def to_spmatrix(self, mode=default_mode, force=False):
        """Returns a sparse numpy matrix representing `self` in matrix form.

        Args:
            mode: str
                Specify the mapping between fermions and qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            numpy sparse matrix
        """
        return self.to_qutip(mode, force).data

    def to_qubit_operator(self, output='qiskit', **kwargs):
        """
        Wrapper function for the conversion of a `BaseFermionOperator` to an operator that acts on qubits.
        The output type depends on the `output` argument as described below.

        Args:
            output: string,
                The desired output type. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']
            **kwargs:
                The keyword arguments to pass to the respective mapping belonging to the `output`

        Returns:
            various types
            The following types are returned depending on the `output` variable:
                'qiskit'   - qiskit.aqua.operators WeightedPauliOperator
                'qutip'    - qutip.Qobj
                'matrix'   - np.ndarray
                'spmatrix' - numpy sparse matrix
        """
        if output == 'qiskit':
            return self.to_qiskit(**kwargs)
        elif output == 'qutip':
            return self.to_qutip(**kwargs)
        elif output == 'matrix':
            return self.to_matrix(**kwargs)
        elif output == 'spmatrix':
            return self.to_spmatrix(**kwargs)
        else:
            supported_output = ['qiskit', 'qutip', 'matrix', 'spmatrix']
            raise TypeError("output type '{}' is not supported. Please use one of {}".format(output, supported_output))
