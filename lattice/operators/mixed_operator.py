import warnings

from .particle_type_operator import ParticleTypeOperator
from .spin_operators import *
from .fermionic_operators import *
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.operators.legacy.op_converter import *
import qutip as qt


def tensor_aqua_operators(aqua_operatorlist):
    """
    Computes the tensor product of a list of qiskit.aqua.operators WeightedPauliOperator objects.
    (I.e. appends the paulis in the order (Paulis of operator 1) @ (Paulis of operator 2) @ ...
    for an operator list [operator1, operator2, ...])

    Args:
        aqua_operatorlist: list
            A list of qiskit.aqua.operators WeightedPauliOperator objects.

    Returns:
        qiskit.aqua.operators WeightedPauliOperator
            The tensor product of the operators (in order specified by the list)
    """

    paulis_per_operator = []

    # Parse input: check if operators are in `paulis` representation and append them to `paulis_per_operator`
    for operator in aqua_operatorlist:
        if not isinstance(operator, WeightedPauliOperator):
            raise TypeError(
                "`aqua_operatorlist` may contain only qiskit.aqua.operators.WeightedPauliOperator objects, "
                "not '{}'".format(operator))
        paulis_per_operator.append(operator.paulis)

    # Generate all possible combinations of tensor products of pauli strings which make up
    # the final tensorproduct operator. (I.e. the summand pauli strings of the final operator)
    combos = list(itertools.product(*paulis_per_operator))
    # Tensor the paulis together
    tensored_paulis = [tensor_paulis(combo) for combo in combos]

    # Construct the final operator as a sum of these pauli strings
    tensorproduct_operator = WeightedPauliOperator(paulis=tensored_paulis)

    return tensorproduct_operator


class GeneralMixedOperator:
    """
    General mixed operators. This class represents sums of mixed operators, i.e. linear combinations of
    MixedOperators with same particle type registers.
    """

    def __init__(self, mixed_operator_list):

        self._registers = mixed_operator_list[0].registers
        self._register_lengths = {}
        for register_type in self._registers:
            self._register_lengths[register_type] = mixed_operator_list[0].register_length(register_type)

        # Check if the elements of the mixed_operator_list are valid & compatible instances of the MixedOperator class
        for mixed_operator in mixed_operator_list:
            assert isinstance(mixed_operator, MixedOperator), 'GeneralMixedOperators must be built up from ' \
                                                                   '`MixedOperator` objects'
            assert np.array_equal(mixed_operator.registers, self._registers), 'GeneralMixedOperator elements must ' \
                                                                              'act on the same particle type ' \
                                                                              'registers in the same order'
            for register_type in self._registers:
                assert mixed_operator.register_length(register_type) == self._register_lengths[register_type], \
                        "Cannot sum '{}' type operators acting on registers of different length".format(register_type)

        # TODO: Find a way to 'factorize' the operator, such that each element only appears once in the operator_list
        self._operator_list = mixed_operator_list

    @property
    def operator_list(self):
        return self._operator_list

    @property
    def registers(self):
        return self._registers

    def register_length(self, register_type):
        assert register_type in self.registers, 'The GeneralMixedOperator does not contain a register ' \
                                                'of type {}'.format(register_type)
        return self._register_lengths[register_type]

    def __repr__(self):
        full_str = 'GeneralMixedOperator acting on registers:'
        for register_name in self.registers:
            full_str += '\n{} :'.ljust(12).format(register_name) + str(self.register_length(register_name))
        full_str += '\nTotal number of MixedOperators:  {}'.format(len(self.operator_list))
        return full_str

    def __add__(self, other):
        """
        Returns a GeneralMixedOperator representing the sum of the given operators

        Args:
            other: MixedOperator or GeneralMixedOperator
                The operator/operatorlist to add to `self`.

        Returns:
            GeneralMixedOperator,
                A `GeneralMixedOperator` object representing the sum of `self` and `other`.
        """

        if isinstance(other, MixedOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            # If the operators are proportional to each other, simply update coefficients
            for idx, operator in enumerate(new_operatorlist):
                is_prop = operator.is_proportional_to(other)
                if is_prop[0]:
                    operator *= (1+is_prop[1])
                    new_operatorlist[idx] = operator
                    return GeneralMixedOperator(new_operatorlist)
            # Else, just append the new operator to the operator_list
            new_operatorlist.append(other)
            return GeneralMixedOperator(new_operatorlist)

        elif isinstance(other, GeneralMixedOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            for elem in other.operator_list:
                new_operatorlist.append(elem)
                # If the operators are proportional to each other, simply update coefficients
                for idx, operator in enumerate(new_operatorlist[:-1]):
                    is_prop = operator.is_proportional_to(elem)
                    if is_prop[0]:
                        new_operatorlist.pop()
                        operator *= (1 + is_prop[1])
                        new_operatorlist[idx] = operator
                        break
                # Else, the new operator has been added to the operator_list
            return GeneralMixedOperator(new_operatorlist)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and 'GeneralMixedOperator'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other):
        """
        Returns a GeneralMixedOperator representing the difference of the given MixedOperators

        Args:
            other: MixedOperator, GeneralMixedOperator
                The Operator to subtract from `self`.

        Returns:
            GeneralMixedOperator,
                A `GeneralMixedOperator` object representing the difference of `self` - `other`.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a number-type."""
        if isinstance(other, numbers.Number):
            # Create copy of the SpinSOperator in which every BaseSpinOperator is multiplied by `other`.
            new_operatorlist = [copy.deepcopy(mixed_operator) * other for mixed_operator in self.operator_list]
            return GeneralMixedOperator(new_operatorlist)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'GeneralMixedOperator' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'GeneralMixedOperator'".format(type(other).__name__))

    def __truediv__(self, other):
        """
        Overloads the division operator `/` for division by number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)
        else:
            raise TypeError(
                "unsupported operand type(s) for /: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def dag(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        daggered_operator_list = [mixed_operator.dag() for mixed_operator in self.operator_list]
        return GeneralMixedOperator(daggered_operator_list)

    def copy(self):
        """
        Copy method. Returns a deepcopy of `self`.

        Returns:
            MixedOperator
        """
        return copy.deepcopy(self)

    def print_operators(self):
        """Print the representations of the operators within the MixedOperator"""
        full_str = 'GeneralMixedOperator\n'

        for operator in self.operator_list:
            full_str += operator.print_operators() + '\n'
        return full_str

    def to_qubit_operator(self,
                          fermion_mapping='jordan_wigner',
                          spin_mapping='log_encoding',
                          output='qiskit'):
        """
        Function for the conversion of a `MixedOperator` to an operator that acts on qubits.
        The output type depends on the `output` argument as described below.
        The order of the tensor product of the register in `self` is given by self.registers()

        Args:
            output (str):
                The desired output type. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']
            fermion_mapping (str):
                The mapping function to be used to map fermions to qubits. Must be one of
                ['jordan_wigner', 'parity', 'bravyi_kitaev']
            spin_mapping (str):
                The mapping function to be used to map spin S systems to qubits. Must be one of
                ['lin_encoding', 'log_encoding']

        Returns:
            various types
            The following types are returned depending on the `output` variable:
                'qiskit'   - qiskit.aqua.Operator
                'qutip'    - qutip.Qobj
                'matrix'   - np.ndarray
                'spmatrix' - numpy sparse matrix
        """
        return operator_sum([operator.to_qubit_operator(fermion_mapping, spin_mapping, output=output)
                             for operator in self.operator_list])


class MixedOperator:
    """
    A class to implement operators that act on different particle type registers.
    Currently supports the following registers: ['fermionic', 'spin 0.5', 'spin 1.0', 'spin 1.5', ...]
    """

    def __init__(self, operator_list):

        self._register_operators = {}
        self._registers = []

        # 1. Parse input and fill the member variables
        assert isinstance(operator_list, (list, np.ndarray)), 'Please provide a list of operators as input.'

        for operator in operator_list:
            if not isinstance(operator, ParticleTypeOperator):
                raise UserWarning("Elements of `operator_list` must be `ParticleTypeOperator` type. Allowed "
                                  "operator types are `FermionicOperator` , `BaseFermionOperator`, "
                                  "`SpinSOperator`, `BaseSpinOperator`.")
            register_name = operator.particle_type
            self[register_name] = operator

    def __getitem__(self, register_name):
        """
       Getter for the individual operators acting on register of different particle types.

       Args:
           register_name (str):
               The name of the register on which the operator acts on. Must be one of ['fermionic', 'bosonic',
               'spin 0.5', 'spin 1.0', ...]

       Returns:
           ParticleTypeOperator:
               The respective ParticleTypeOperator.
        """
        # Check for correct indexing
        self._is_allowed_key(register_name)

        return self._register_operators[register_name]

    def __setitem__(self, register_name, operator):
        """
        Setter for the individual operators acting on register of different particle types.

        Args:
            register_name (str):
                The name of the register on which the operator acts on. Must be one of ['fermionic', 'bosonic',
                'spin 0.5', 'spin 1.0', ...]
            operator (ParticleTypeOperator):
                A ParticleTypeOperator object representing the operator on the specific register

        Returns:
            None
        """
        # 1. Parse
        #  Check for correct indexing
        self._is_allowed_key(register_name)
        #  Check if the given operator matches the register_name.
        if not register_name == operator.particle_type:
            raise UserWarning("Cannot assign a '{}' type operator to a '{}' register ".format(operator.particle_type,
                                                                                              register_name))
        # 2. Warn if an operator will be overwritten
        if register_name in self.registers:
            warnings.warn("MixedOperator already has a '{}' register. Setting it overwrites it.".format(register_name))
        else:
            self._registers.append(register_name)

        # 3. Assign the operator to the respective register
        self._register_operators[register_name] = operator

    def __matmul__(self, other):
        """Implements the operator tensorproduct"""
        if isinstance(other, ParticleTypeOperator):
            new_mixed_operator = copy.deepcopy(self)
            assert other.particle_type not in new_mixed_operator.registers, \
                "Operator already has a '{0}' register. Please include all '{0}' operators " \
                "into this register.".format(other.particle_type)
            new_mixed_operator[other.particle_type] = other

        elif isinstance(other, MixedOperator):
            new_mixed_operator = copy.deepcopy(self)
            for register_name in other.registers:
                assert register_name not in new_mixed_operator.registers, \
                    "Operator already has a '{0}' register. Please include all '{0}' operators " \
                    "into this register.".format(register_name)
                new_mixed_operator[register_name] = other[register_name]

        else:
            raise TypeError("unsupported operand @ for objects of type '{}' and '{}'".format(type(self).__name__,
                                                                                             type(other).__name__))
        return new_mixed_operator

    def __add__(self, other):
        """
        Returns a GeneralMixedOperator representing the sum of the given operators

        Args:
            other: MixedOperator or GeneralMixedOperator
                The operator/operatorlist to add to `self`.

        Returns:
            GeneralMixedOperator,
                A `GeneralMixedOperator` object representing the sum of `self` and `other`.
        """
        if isinstance(other, MixedOperator):
            is_prop = self.is_proportional_to(other)
            if is_prop[0]:
                return self.__mul__(other=(1+is_prop[1]))
            else:
                return GeneralMixedOperator([self.copy(), other.copy()])

        elif isinstance(other, GeneralMixedOperator):
            return other.__add__(self)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and 'MixedOperator'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other):
        """
        Returns a GeneralMixedOperator representing the difference of the given MixedOperators

        Args:
            other: MixedOperator, GeneralMixedOperator
                The Operator to subtract from `self`.

        Returns:
            GeneralMixedOperator,
                A `GeneralMixedOperator` object representing the difference of `self` - `other`.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a number-type."""
        if isinstance(other, numbers.Number):
            # Absorb the multiplication factor into the first register (could also be absorbed in any other register)
            first_register_type = self.registers[0]
            new_mixed_operator = copy.deepcopy(self)
            # Catch the warning (from MixedOperator.__setitem__(...)) when a register is being updated.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="MixedOperator already has a '{}' register. "
                                                          "Setting it overwrites it.".format(first_register_type))
                new_mixed_operator[first_register_type] *= other
            return new_mixed_operator
        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'MixedOperator' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'MixedOperator'".format(type(other).__name__))

    def __truediv__(self, other):
        """
        Overloads the division operator `/` for division by number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)
        else:
            raise TypeError(
                "unsupported operand type(s) for /: 'FermionicOperator' and '{}'".format(type(other).__name__))

    def dag(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        daggered_operator_list = [self[register_type].dag() for register_type in self.registers]
        return MixedOperator(daggered_operator_list)

    def __repr__(self):
        full_str = 'MixedOperator acting on registers:'
        for register_name in self.registers:
            full_str += '\n{} :'.ljust(12).format(register_name) + str(self.register_length(register_name))
        return full_str

    def copy(self):
        """
        Copy method. Returns a deepcopy of `self`.

        Returns:
            MixedOperator
        """
        return copy.deepcopy(self)

    @property
    def registers(self):
        """
        Return the particle types that MixedOperator `self` acts on. The list order
        corresponds to the order of the tensor product.

        Returns:
            list
                The list of registers of differnt particle type that the MixedOperator acts on. This order is
                also the tensor product order.
        """
        return self._registers

    def print_operators(self):
        """Print the representations of the operators within the MixedOperator"""
        full_str = 'MixedOperator\n'

        for register in self.registers:
            full_str += (register + ': \n') + self[register].__repr__() + '\n'
        return full_str

    def register_length(self, register_name):
        """
        Returns the length of the register with name `register_name`.

        Args:
            register_name (str):
               The name of the register on which the operator acts on. Must be one of ['fermionic', 'bosonic',
               'spin 0.5', 'spin 1.0', ...]

        Returns:
            int
        """
        # Check for correct indexing
        self._is_allowed_key(register_name)
        # Return length of the respective register
        return self[register_name].register_length

    @staticmethod
    def _is_allowed_key(key):
        """
        Checks whether `key` is an allowed key, i.e. one of ['fermionic', 'bosonic',
               'spin 0.5', 'spin 1.0', ...]
        Args:
            key (str):
                Must be one of ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', 'spin 1.5', ...]
        Returns:
            bool
        """
        if key == 'fermionic':
            return True
        elif key == 'bosonic':
            raise NotImplementedError('Not implemented yet. Currently only supports fermionic an spin operators')
        elif key[0:4] == 'spin':
            S = float(key[5:])
            if (scipy.fix(2 * S) != 2 * S) or (S < 0):
                raise TypeError('S must be a non-negative integer or half-integer')
            else:
                return True
        else:
            raise UserWarning("Allowed register arguments are ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', ...]"
                              " not '{}'").format(key)

    def is_proportional_to(self, other):
        """
        Checks whether two MixedOperators (M1, M2) are proportional to each other, c * M1 = M2,
        where c is a complex number and M1 = `self` and M2 = `other`. (Used for adding two MixedOperator type objects)

        Args:
            other (MixedOperator):
                Must be a MixedOperator
        Returns:
            list
                Returns a list [bool, numbers.Number] with the corresponding factor of proportionality
        """

        # Parse for validity and compatibility
        assert isinstance(other, MixedOperator), '`other` must be a `MixedOperator` type object'
        assert np.array_equal(other.registers, self.registers), 'The two MixedOperators must act on the same ' \
                                                                'particle type registers in the same order'
        for register_type in self.registers:
            assert other.register_length(register_type) == self.register_length(register_type), \
                    "Cannot compare '{}' type operators acting on registers of different length".format(register_type)

        # Check for proportionality and calculate the corresponding factor
        factor = 1.  # Define factor of proportionality
        for register_type in self.registers:

            # 0. Convert BaseFermionOperators to FermionicOperators and BaseSpinOperators to SpinSOperators
            if isinstance(self[register_type], BaseFermionOperator):
                register_1 = copy.deepcopy(self[register_type])
                register_1 = FermionicOperator([register_1])
            elif isinstance(self[register_type], BaseSpinOperator):
                register_1 = copy.deepcopy(self[register_type])
                register_1 = SpinSOperator([register_1])
            else:
                register_1 = self[register_type]

            if isinstance(other[register_type], BaseFermionOperator):
                register_2 = copy.deepcopy(other[register_type])
                register_2 = FermionicOperator([register_2])
            elif isinstance(other[register_type], BaseSpinOperator):
                register_2 = copy.deepcopy(other[register_type])
                register_2 = SpinSOperator([register_2])
            else:
                register_2 = other[register_type]

            # 1. Generate dictionaries for the particle type operators
            operator_dict_1 = {}
            operator_dict_2 = {}
            for op1 in register_1.operator_list:
                operator_dict_1[op1.label] = op1.coeff
            for op2 in register_2.operator_list:
                operator_dict_2[op2.label] = op2.coeff

            # 2. Check if all labels of the MixedOperators are equal
            label_set_1 = set(operator_dict_1.keys())
            label_set_2 = set(operator_dict_2.keys())
            # 2.1 Check if the two label sets are equal
            if bool(label_set_1.symmetric_difference(label_set_2)):  # bool(...) returns `False` iff the set is empty
                return [False, None]

            # 3. Check for proportionality
            # Set a reference label and coefficient and `normalize` the other coefficients according to the reference,
            # in order to compare the operators
            ref_label = list(operator_dict_1.keys())[0]
            ref_coeff_1 = operator_dict_1[ref_label]
            ref_coeff_2 = operator_dict_2[ref_label]
            for op_label in operator_dict_1.keys():
                coeff_1 = operator_dict_1[op_label] / ref_coeff_1
                coeff_2 = operator_dict_2[op_label] / ref_coeff_2
                # if not np.allclose([coeff_1], [coeff_2]):
                if coeff_1 != coeff_2:
                    return [False, None]

            # 4. Update factor of proportionality
            factor *= (ref_coeff_2 / ref_coeff_1)

        return [True, factor]

    def to_qubit_operator(self,
                          fermion_mapping='jordan_wigner',
                          spin_mapping='log_encoding',
                          output='qiskit'):
        """
        Function for the conversion of a `MixedOperator` to an operator that acts on qubits.
        The output type depends on the `output` argument as described below.
        The order of the tensor product of the register in `self` is given by self.registers()

        Args:
            output (str):
                The desired output type. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']
            fermion_mapping (str):
                The mapping function to be used to map fermions to qubits. Must be one of
                ['jordan_wigner', 'parity', 'bravyi_kitaev']
            spin_mapping (str):
                The mapping function to be used to map spin S systems to qubits. Must be one of
                ['lin_encoding', 'log_encoding']
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

        # output: can be one of `qiskit`, `qutip`, `matrix`, `spmatrix`
        # fermion_mapping can be one of `jordan-wigner`

        # 0. Set up supported mappings and outputs
        supported_fermion_mappings = ['jordan_wigner', 'parity', 'bravyi_kitaev']
        supported_spin_mappings = ['lin_encoding', 'log_encoding']
        supported_outputs = ['qiskit', 'qutip', 'matrix', 'spmatrix']

        # 1. Parse mappings and output
        #  check for supported mappings
        if not fermion_mapping in supported_fermion_mappings:
            raise UserWarning(
                "Fermion mapping '{}' not supported. `fermion_mapping` must be "
                "one of {}".format(fermion_mapping, supported_fermion_mappings))
        if not spin_mapping in supported_spin_mappings:
            raise UserWarning("Spin mapping '{}' not supported. `spin_mapping` "
                              "must be one of {}".format(spin_mapping, supported_spin_mappings))
        #  check for supported output
        if not output in supported_outputs:
            raise UserWarning("Output '{}' not supported. `output` must be one of {}".format(output, supported_outputs))

        # 2. Transform each particle type to the respective qubit operator in output form with the requested transform
        #    specified for this particle type
        # Set up a list to store all qubit operators
        qubit_operators = []
        # Set up temporary output (if qiskit: qiskit, if matrix/spmatrix/qutip: qutip)
        temp_output = 'qiskit' if output == 'qiskit' else 'qutip'
        # Transform the particle type operator to qubits
        for register_name in self.registers:
            if register_name == 'fermionic':
                qubit_operator = self[register_name].to_qubit_operator(output=temp_output, mode=fermion_mapping)
            else:
                qubit_operator = self[register_name].to_qubit_operator(output=temp_output, mode=spin_mapping)
            qubit_operators.append(qubit_operator)

        # 3. Tensor together the individual qubit operators
        # Tensor together the operators
        if output == 'qiskit':
            return tensor_aqua_operators(qubit_operators)
        else:
            tensored_operator = qt.tensor(qubit_operators[::-1])
            # Again, the [::-1] is because qutip and qiskit have the reversed tensor product ordering.

            if output == 'matrix':
                # dense matrix case
                return tensored_operator.data.todense()
            elif output == 'spmatrix':
                # sparse matrix case
                return tensored_operator.data
            else:
                # qutip case
                return tensored_operator
