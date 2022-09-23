import copy
import itertools
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.operators.legacy.op_converter import *
import numbers
import numpy as np
import qutip as qt
import scipy
from .qiskit_aqua_operator_utils import *
from .particle_type_operator import ParticleTypeOperator

default_spin_mode = 'log_encoding'


def embed_matrix(matrix, nqubits, embed_padding=0., embed_location='upper'):
    """
    Embeds `matrix` into the upper/lower diagonal block of a 2^nqubits by 2^nqubits matrix and pads the
    diagonal of the upper left block matrix with the value of `embed_padding`. Whether the upper/lower
    diagonal block is used depends on `embed_location`.
    I.e. using embed_location = 'upper' returns the matrix:
        [[ matrix,    0             ],
         [   0   , embed_padding * I]]

    Using embed_location = 'lower' returns the matrix:
        [[ embed_padding * I,    0    ],
         [      0           ,  matrix ]]


    Args:
        matrix (numpy.ndarray):
            The matrix (2D-array) to embed

        nqubits (int):
            The number of qubits on which the embedded matrix should act on.

        embed_padding (float):
            The value of the diagonal elements of the upper left block of the embedded matrix.

        embed_location (str):
            Must be one of ['upper', 'lower']. This parameters sets whether the given matrix is embedded in the
            upper left hand corner or the lower right hand corner of the larger matrix.

    Returns:
        full_matrix (numpy.ndarray):
            If `matrix` is of size 2^nqubits, returns `matrix`.
            Else it returns the block matrix (I = identity)
            [[ embed_padding * I,    0    ],
             [      0           , `matrix`]]
    """
    full_dim = 1 << nqubits
    subs_dim = matrix.shape[0]

    dim_diff = full_dim - subs_dim
    if dim_diff == 0:
        return matrix

    elif dim_diff > 0:
        if embed_location == 'lower':
            full_matrix = np.zeros((full_dim, full_dim), dtype=complex)
            full_matrix[:dim_diff, :dim_diff] = np.eye(dim_diff) * embed_padding
            full_matrix[dim_diff:, dim_diff:] = matrix

        elif embed_location == 'upper':
            full_matrix = np.zeros((full_dim, full_dim), dtype=complex)
            full_matrix[:subs_dim, :subs_dim] = matrix
            full_matrix[subs_dim:, subs_dim:] = np.eye(dim_diff) * embed_padding

        else:
            raise UserWarning('embed_location must be one of ["upper","lower"]')

        return full_matrix

    else:
        raise UserWarning('The given matrix does not fit into the space spanned by {} qubits'.format(nqubits))


def embed_state(vector, nqubits, embed_location='upper'):
    """
    Embeds the statevector `vector` into the upper/lower part of a 2^nqubits statevector and pads
    the remaining values with 0.
    I.e. using embed_location = 'upper' returns the vector:
         [--- state vector ---, 0., ... , 0., 0.],

    Using embed_location = 'lower' returns the vector:
         [ 0., 0., ... , --- state vector ---],
         
    Args:
        vector (np.ndarray):
            The state vector to embed

        nqubits (int):
            The number of qubits in which the given statevector should be encoded.

        embed_location (str):
            Must be one of ['upper', 'lower']. This parameters sets whether the given vector is embedded in the
            upper part or the lower part of the larger statevector.

    Returns:
        full_vector (np.ndarray):
            If `vector` is of size 2^nqubits, returns `vector`.
            Else it returns the np.ndarray
            [--- state vector ---, 0., ... , 0., 0.],
    """
    full_dim = 1 << nqubits
    vector = np.asarray(vector)
    subs_dim = vector.shape[0]
    if not np.isclose(np.linalg.norm(vector), 1.):
        raise UserWarning('The given state vector is not normalized.')

    dim_diff = full_dim - subs_dim
    if dim_diff == 0:
        return vector
    
    elif dim_diff > 0:
        if embed_location == 'lower':
            full_vector = np.zeros(full_dim, dtype=complex)
            full_vector[dim_diff:] = vector
        elif embed_location == 'upper':
            full_vector = np.zeros(full_dim, dtype=complex)
            full_vector[:subs_dim] = vector
        else:
            raise UserWarning('embed_location must be one of ["upper","lower"]')

        return full_vector

    else:
        raise UserWarning('The given state vector does not fit into the space spanned by {} qubits'.format(nqubits))


class SpinSOperator(ParticleTypeOperator):
    """
    Spin S type operators. This class represents sums of `spin strings`, i.e. linear combinations of
    BaseSpinOperators with same S and register length.
    """

    def __init__(self, operator_list):
        # 1. Parse input
        self.register_length = len(operator_list[0])
        self.S = operator_list[0].S
        self.transformed_XYZI = operator_list[0].transformed_XYZI
        for elem in operator_list:
            assert isinstance(elem, BaseSpinOperator)
            assert len(elem) == self.register_length, 'Cannot sum operators acting on registers of different length'
            assert elem.S == self.S, 'Cannot sum operators corresponding to different spin `S`.'
            # Initialize the XYZI transform dictionary which transforms spin operators to qubits
            if self.transformed_XYZI is None:
                self.transformed_XYZI = elem.transformed_XYZI


        # 2. Initialize the operator list of `self`
        self._operator_list = operator_list

        # 3. Set the operators particle type to 'spin S' with S the spin value (as float with 1 decimal).
        ParticleTypeOperator.__init__(self, particle_type='spin {0:.1f}'.format(self.S))

    def __repr__(self):
        full_str = ''
        for operator in self._operator_list:
            full_str += '{1} \t {0}\n'.format(operator.coeff, operator.label)
        return full_str

    def __add__(self, other):  # TODO: Make this much more efficient by working with lists and label indices
        """
        Returns a SpinOperator representing the sum of the given operators

        Args:
            other: BaseSpinOperator or SpinSOperator
                The operator/operatorlist to add to `self`.

        Returns:
            SpinOperator,
                A `SpinOperator` object representing the sum of `self` and `other`.
        """

        if isinstance(other, BaseSpinOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            # If the operators have the same label, simply add coefficients:
            for operator in new_operatorlist:
                if other.label == operator.label:
                    sum_coeff = operator.coeff + other.coeff
                    operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                    # if the new coefficient is zero, remove the operator from the list
                    if sum_coeff == 0:
                        new_operatorlist.remove(operator)
                    return SpinSOperator(new_operatorlist)
            new_operatorlist.append(other)
            return SpinSOperator(new_operatorlist)

        elif isinstance(other, SpinSOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            for elem in other.operator_list:
                new_operatorlist.append(elem)
                # If the operators have the same label, simply add coefficients:
                for operator in new_operatorlist[:-1]:
                    if elem.label == operator.label:
                        new_operatorlist.pop()
                        sum_coeff = operator.coeff + elem.coeff
                        operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                        # if the new coefficient is zero, remove the operator from the list
                        if sum_coeff == 0:
                            new_operatorlist.remove(operator)
                        break
            return SpinSOperator(new_operatorlist)

        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and 'SpinOperator'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other):
        """
        Returns a SpinOperator representing the difference of the given BaseSpinOperators

        Args:
            other: BaseSpinOperator, SpinSOperator
                The Operator to subtract from `self`.

        Returns:
            SpinOperator,
                A `SpinOperator` object representing the difference of `self` - `other`.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a number-type."""
        if isinstance(other, numbers.Number):
            # Create copy of the SpinSOperator in which every BaseSpinOperator is multiplied by `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other for base_operator in self.operator_list]
            return SpinSOperator(new_operatorlist)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'SpinSOperator' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'SpinSOperator'".format(type(other).__name__))

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
    def operator_list(self):
        return self._operator_list

    def dag(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        daggered_operator_list = [operator.dag() for operator in self.operator_list]
        return SpinSOperator(daggered_operator_list)

    def to_qutip(self, mode='lin_encoding', force=False):
        """
        Returns the qutip.Quobj which represents the given SpinSOperator `self`.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the qutip.Qobj is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            qutip.Qobj
        """
        return operator_sum([operator.to_qutip(mode, force) for operator in self.operator_list])

    def to_matrix(self, mode='lin_encoding', force=False):
        """Returns a dense numpy matrix representing `self` in matrix form.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            np.ndarray
        """
        return sum([operator.to_matrix(mode, force) for operator in self.operator_list])

    def to_spmatrix(self, mode='lin_encoding', force=False):
        """Returns a sparse numpy matrix representing `self` in matrix form.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            numpy sparse matrix
        """
        return sum([operator.to_spmatrix(mode, force) for operator in self.operator_list])

    def to_qiskit(self, mode = default_spin_mode, precomputed_XYZI_transform=None):
        """
        Creates a qiskit.aqua.operators WeightedPauliOperator from `self` that maps the spinS system operator
        to a qubit operator.

        Args:
            precomputed_XYZI_transform: list
                This list must contain exactly 4 elements corresponding to the transformations for the X, Y, Z and
                identity operators. Each of these 4 elements must be a qiskit.aqua.operators WeightedPauliOperator
                (i.e. a list of Pauli strings).

        Returns:
            qiskit.aqua.operators WeightedPauliOperator
        """
        # 1. Check if there is a precomputed transform present for the object or given as argument
        #    and set the trafo_to_use accordingly.
        if self.transformed_XYZI is None and precomputed_XYZI_transform is None:
            # if neither is given, calculate the specified transform and store it in self.transformed_XYZI
            if mode == 'log_encoding':
                self.transformed_XYZI = self.operator_list[0]._logarithmic_encoding()
            elif mode == 'lin_encoding':
                self.transformed_XYZI = self.operator_list[0]._linear_encoding()
            else:
                raise UserWarning("`mode` for spinS to qubit transform must be one of "
                                  "['log_encoding', lin_encoding'].")
            transform_to_use = self.transformed_XYZI

        elif precomputed_XYZI_transform is not None:
            # if there's a precomputed transform given, use that one
            transform_to_use = precomputed_XYZI_transform

        elif self.transformed_XYZI is not None:
            transform_to_use = self.transformed_XYZI

        # 2. Perform the transformation on each individual operator in the SpinSOperator and return the aqua Operator
        return operator_sum([operator.to_qiskit(precomputed_XYZI_transform=transform_to_use)
                             for operator in self.operator_list])

    def to_qubit_operator(self, output='qiskit', **kwargs):
        """
        Wrapper function for the conversion of a `SpinSOperator` to an operator that acts on qubits.
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


class BaseSpinOperator(ParticleTypeOperator):
    """
    A class for products and powers of XYZ-ordered SpinS operators.
    """

    def __init__(self, S, Sx, Sy, Sz, coeff=1.):

        # 1. Infer the number of individual spin systems in the register
        self.register_length = len(Sx)

        # 2. Parse input
        # Parse S
        if (scipy.fix(2 * S) != 2 * S) or (S < 0):
            raise TypeError('S must be a non-negative integer or half-integer')

        # Parse coeff
        if not isinstance(coeff, numbers.Number):
            raise TypeError("`coeff` must be a number type not '{}'".format(type(coeff).__name__))

        for spin_operators in [Sx, Sy, Sz]:
            # Check the input type (arrays)
            if not isinstance(spin_operators, (list, np.ndarray)):
                raise TypeError(
                    "Sx, Sy and Sz must be 'np.ndarray' with integers, not '{}'".format(type(spin_operators).__name__))

            # Check the length
            assert len(spin_operators) == self.register_length, "`Sx, Sy, Sz` must be of same length."

            # Check datatype of first elements
            if not isinstance(spin_operators[0], (int, np.integer)):
                raise TypeError("Elements of `Sx, Sy, Sz` must be of integer type.")

        # 3. Initialize the member variables
        self._S = S
        self._coeff = coeff
        self._Sx = np.asarray(Sx).astype(dtype=np.uint16, copy=True)
        self._Sy = np.asarray(Sy).astype(dtype=np.uint16, copy=True)
        self._Sz = np.asarray(Sz).astype(dtype=np.uint16, copy=True)
        self.transformed_XYZI = None
        self._label = None
        self.generate_label()

        # 4. Set the operators particle type to 'spin S' with S the spin value (as float with 1 decimal).
        ParticleTypeOperator.__init__(self, particle_type='spin {0:.1f}'.format(self.S))

    @property
    def S(self):
        """The spin value S of the individual spin systems in the register. The dimension of the
        spin systems is therefore 2S+1."""
        return self._S

    @property
    def coeff(self):
        """The (complex) coefficient of the spin operator."""
        return self._coeff

    @property
    def Sx(self):
        """A np.ndarray storing the power i of (spin) X operators on the spin system.
        I.e.
            [0, 4, 2] corresponds to X0^0 \otimes X1^4 \otimes X2^2, where Xi acts on the i-th spin system
            in the register.
        """
        return self._Sx

    @property
    def Sy(self):
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e.
            [0, 4, 2] corresponds to Y0^0 \otimes Y1^4 \otimes Y2^2, where Yi acts on the i-th spin system
            in the register.
        """
        return self._Sy

    @property
    def Sz(self):
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e.
            [0, 4, 2] corresponds to Z0^0 \otimes Z1^4 \otimes Z2^2, where Zi acts on the i-th spin system
            in the register.
        """
        return self._Sz

    @property
    def label(self):
        """The description of `self` in terms of a string label."""
        return self._label

    def __len__(self) -> int:
        """
        Returns the number of spin systems in the spin register, i.e. the length of `self.Sx` (or Sy, Sz)
        """
        return self.register_length

    def __repr__(self) -> str:
        """
        Prints `self.coeff` and `self.label` to the console.
        """
        return self.label + ' \t ' + str(self.coeff)

    def __eq__(self, other):
        """Overload == ."""
        if not isinstance(other, BaseSpinOperator):
            return False

        S_equals = (self.S == other.S)
        Sx_equals = np.all(self.Sx == other.Sx)
        Sy_equals = np.all(self.Sy == other.Sy)
        Sz_equals = np.all(self.Sz == other.Sz)
        coeff_equals = np.all(self.coeff == other.coeff)

        return S_equals and Sx_equals and Sy_equals and Sz_equals and coeff_equals

    def __ne__(self, other):
        """Overload != ."""
        return not self.__eq__(other)

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def generate_label(self):
        """Generates the string description of `self`."""
        label = ''
        for pos, nx, ny, nz in zip(np.arange(self.register_length), self.Sx, self.Sy, self.Sz):
            if nx > 0:
                label += ' X^{}'.format(nx)
            if ny > 0:
                label += ' Y^{}'.format(ny)
            if nz > 0:
                label += ' Z^{}'.format(nz)
            if nx > 0 or ny > 0 or nz > 0:
                label += '[{}] |'.format(pos)
            else:
                label += ' I[{}] |'.format(pos)

        self._label = label[1:-2]  # remove leading and traling whitespaces and trailing |
        return self.label

    def __add__(self, other):
        """
        Returns a SpinOperator representing the sum of the given BaseSpinOperators

        Args:
            other: BaseSpinOperator,
                The BaseSpinOperator to add to `self`.

        Returns:
            SpinOperator,
                A `SpinOperator` object representing the sum of `self` and `other`.
        """

        if isinstance(other, BaseSpinOperator):
            # If the operators have the same label, simply add coefficients:
            if other.label == self.label:
                sum_coeff = self.coeff + other.coeff
                sum_operator = copy.deepcopy(self)  # create a copy of the initial operator to preserve initialize trafos
                sum_operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                return sum_operator
            else:
                return SpinSOperator([copy.deepcopy(self), copy.deepcopy(other)])
        elif isinstance(other, SpinSOperator):
            #  In this case use the __add__ method of FermionicOperator.
            return other.__add__(self)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and 'BaseSpinOperator'".format(type(other).__name__))

    def __sub__(self, other):
        """
        Returns a SpinOperator representing the difference of the given BaseSpinOperators

        Args:
            other: BaseSpinOperator,
                The BaseSpinOperator to subtract from `self`.

        Returns:
            SpinOperator,
                A `SpinOperator` object representing the difference of `self` - `other`.
        """
        return self.__add__((-1) * other)

    def __rmul__(self, other):
        """
        Overloads the right multiplication operator `*` for multiplication with number-type objects
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'BaseSpinOperator'".format(type(other).__name__))

    def __mul__(self, other):
        """
        Overloads the multiplication operator `*` for self and other, where other is a number-type object.
        """
        if isinstance(other, numbers.Number):
            product_operator = copy.deepcopy(self)  # create a copy of self (to also preserve pre-computed transforms)
            product_operator._coeff *= other
            return product_operator

        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and 'BaseSpinOperator'".format(type(other).__name__))

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
        # Note: X, Y, Z are hermitean, therefore the dagger operation on a BaseSpinOperator amounts
        #         # to simply complex conjugating the coefficient.
        new_operator = copy.deepcopy(self)  # create a copy of self (to also preserve pre-computed transforms)
        new_operator._coeff = np.conj(self.coeff)
        return new_operator

    def to_matrix(self, mode='lin_encoding', force=False):
        """Returns a dense numpy matrix representing `self` in matrix form.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            np.ndarray
        """
        return self.to_qutip(mode, force).data.todense()

    def to_spmatrix(self, mode='lin_encoding', force=False):
        """Returns a sparse numpy matrix representing `self` in matrix form.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the matrix is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            numpy sparse matrix
        """
        return self.to_qutip(mode, force).data

    def to_qutip(self, mode='lin_encoding', force=False):
        """
        Returns the qutip.Quobj which represents the given BaseSpinOperator `self`.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits.
            force: bool
                if True, the creation of the qutip.Qobj is forced to proceed even if the resulting matrix is
                very large. If False, only matrices of sizes < 1e5 will be created.

        Returns:
            qutip.Qobj
        """

        if mode == 'lin_encoding':
            dim_S = int(2 * self.S + 1)
            expected_dim = dim_S ** self.register_length
            if expected_dim > 1e5 and not force:
                raise UserWarning(
                    'Warning! Expected matrix dimension is {}. Creation will take a significant time. To proceed, '
                    'set argument `force` to True.'.format(expected_dim))

            operatorlist = []

            spinx = qt.jmat(self.S, 'x')
            spiny = qt.jmat(self.S, 'y')
            spinz = qt.jmat(self.S, 'z')
            identity = qt.identity(dim_S)

            # TODO: Warning: The [::-1] is just needed to make it agree with the QISKIT ordering.
            # If I use [::-1], here, I also need to construct Qutip states via [::-1].
            for nx, ny, nz in zip(self.Sx[::-1], self.Sy[::-1], self.Sz[::-1]):

                operator_on_spin_i = []
                if nx > 0:
                    operator_on_spin_i.append(spinx ** nx)
                if ny > 0:
                    operator_on_spin_i.append(spiny ** ny)
                if nz > 0:
                    operator_on_spin_i.append(spinz ** nz)
                if np.any([nx, ny, nz]) > 0:
                    operatorlist.append(operator_product(operator_on_spin_i))
                else:
                    operatorlist.append(identity)
         ################ l'ho spostato us ######################           
            return self.coeff * qt.tensor(operatorlist)
        ################# aggiungo anche il log encoding usando to _ qiskit direttamnte !     
        elif mode == "log_encoding":
            ## weighted pauli operator 
            op_log_qiskit =self.to_qiskit(mode="log_encoding")
            op_log_qutip = qt.Qobj(to_matrix_operator(op_log_qiskit).dense_matrix)
            return op_log_qutip
        else:
            raise UserWarning("`mode` for BaseSpinOperator to qubit transform must be one of ['lin_encoding'] or log encoding.")

        

    def _logarithmic_encoding(self, embed_padding=0., embed_location='upper'):
        """
        Generates a 'local_encoding_transformation' of the spin S operators 'X', 'Y', 'Z' and 'identity'
        to qubit operators (linear combinations of pauli strings).
        In this 'local_encoding_transformation' each individual spin S system is represented via
        the lowest lying 2S+1 states in a qubit system with the minimal number of qubits needed to
        represent >= 2S+1 distinct states.

        Args:
            embed_padding: complex,
                The matrix element to which the diagonal matrix elements for the 2^nqubits - (2S+1) 'unphysical'
                states should be set to.

        Returns:
            self.transformed_XYZI: list,
                The 4-element list of transformed spin S 'X', 'Y', 'Z' and 'identity' operators.
                I.e.
                    self.transformed_XYZI[0] corresponds to the linear combination of pauli strings needed
                    to represent the embedded 'X' operator
        """
        print('Log encoding is calculated.')
        self.transformed_XYZI = []
        dim_S = int(2 * self.S + 1)
        nqubits = int(np.ceil(np.log2(dim_S)))

        # Get the spin matrices (from qutip)
        spin_matrices = [np.asarray(qt.jmat(self.S, symbol).data.todense()) for symbol in 'xyz']
        # Append the identity
        spin_matrices.append(np.eye(dim_S))

        # Embed the spin matrices in a larger matrix of size 2**nqubits x 2**nqubits
        embed = lambda matrix: embed_matrix(matrix,
                                            nqubits,
                                            embed_padding=embed_padding,
                                            embed_location=embed_location)
        embedded_spin_matrices = list(map(embed, spin_matrices))

        # Generate aqua operators from these embeded spin matrices to then perform the Pauli-Scalar product
        embedded_aqua_operators = [MatrixOperator(matrix=matrix) for matrix in embedded_spin_matrices]
        # Perform the projections onto the pauli strings via the scalar product:
        for op in embedded_aqua_operators:
            op = to_weighted_pauli_operator(op)
            op.chop()
            self.transformed_XYZI.append(op)
        return self.transformed_XYZI

    def _linear_encoding(self):
        """
        Generates a 'linear_encoding' of the spin S operators 'X', 'Y', 'Z' and 'identity'
        to qubit operators (linear combinations of pauli strings).
        In this 'linear_encoding' each individual spin S system is represented via
        2S+1 qubits and the state |s> is mapped to the state |00...010..00>, where the s-th qubit is
        in state 1.

        Returns:
            self.transformed_XYZI: list,
                The 4-element list of transformed spin S 'X', 'Y', 'Z' and 'identity' operators.
                I.e.
                    self.transformed_XYZI[0] corresponds to the linear combination of pauli strings needed
                    to represent the embedded 'X' operator
        """
        print('Linear encoding is calculated.')
        self.transformed_XYZI = []
        dim_S = int(2 * self.S + 1)
        nqubits = dim_S

        # quick functions to generate a pauli with X / Y / Z at location `i`
        pauli_id = Pauli.from_label('I' * nqubits)
        pauli_x = lambda i: Pauli.from_label('I' * i + 'X' + 'I' * (nqubits - i - 1))
        pauli_y = lambda i: Pauli.from_label('I' * i + 'Y' + 'I' * (nqubits - i - 1))
        pauli_z = lambda i: Pauli.from_label('I' * i + 'Z' + 'I' * (nqubits - i - 1))

        # 1. build the non-diagonal X operator
        x_summands = []
        for i, coeff in enumerate(np.diag(qt.jmat(self.S, 'x'), 1)):
            x_summands.append(WeightedPauliOperator(paulis=[[coeff / 2., pauli_x(i) * pauli_x(i + 1)],
                                               [coeff / 2., pauli_y(i) * pauli_y(i + 1)]])
                              )
        self.transformed_XYZI.append(operator_sum(x_summands))

        # 2. build the non-diagonal Y operator
        y_summands = []
        for i, coeff in enumerate(np.diag(qt.jmat(self.S, 'y'), 1)):
            y_summands.append(WeightedPauliOperator(paulis=[[-1j * coeff / 2., pauli_x(i) * pauli_y(i + 1)],
                                               [1j * coeff / 2., pauli_y(i) * pauli_x(i + 1)]])
                              )
        self.transformed_XYZI.append(operator_sum(y_summands))
        
        # 3. build the diagonal Z
        z_summands = []
        for i, coeff in enumerate(np.diag(qt.jmat(self.S, 'z'))):  # get the first upper diagonal of coeff.
            z_summands.append(WeightedPauliOperator(paulis=[[coeff/2., pauli_z(i)],
                                               [coeff/2., pauli_id]])
                              )
        z_operator = operator_sum(z_summands)
        z_operator.chop()
        self.transformed_XYZI.append(z_operator)

        # 4. add the identity operator
        self.transformed_XYZI.append(WeightedPauliOperator(paulis=[[1., pauli_id]]))

        # return the lookup table for the transformed XYZI operators
        return self.transformed_XYZI

    def to_qiskit(self, mode=default_spin_mode, precomputed_XYZI_transform=None):
        """
        Creates a qiskit.aqua.operators.WeightedPauliOperator from `self` that maps the spinS
        system operator to a qubit operator.

        Args:
            mode (str):
                The mode with which the spin operators should be mapped to qubits. Should be one of
                ['lin_encoding', 'log_encoding'].

            precomputed_XYZI_transform (list):
                This list must contain exactly 4 elements corresponding to the transformations for the X, Y, Z and
                the identity operator. Each of these 4 elements must be a qiskit.aqua.operators.WeightedPauliOperator
                in `pauli` representation (i.e. a list of Pauli strings)

        Returns:
            qiskit.aqua.operators.WeightedPauliOperator
        """
        # 1. Setup
        # Get the dimension of the internal spin Hilbertspace per spin system
        dim_S = int(2 * self.S + 1)

        # Initialize an empty list, used to store the embedded qubit operators
        # used to construct the final aqua operator.
        operatorlist = []

        # 2. Generate the mapping (spin S system --> qubits)
        # Map the X, Y, Z operators of an individual spin system to the `nqubit` system.
        # The final operator will be the sum of a tensor product of these operators with
        # complex coefficients.
        if self.transformed_XYZI is None and precomputed_XYZI_transform is None:
            # if neither is given, calculate a specified transform and store it in self.transformed_XYZI
            print('Calculating encoding for operator')
            if mode == 'log_encoding':
                self.transformed_XYZI = self._logarithmic_encoding()
            elif mode == 'lin_encoding':
                self.transformed_XYZI = self._linear_encoding()
            else:
                raise UserWarning("`mode` for BaseSpinOperator to qubit transform must be one of "
                                  "['log_encoding', lin_encoding'].")

        if precomputed_XYZI_transform is None:
            spinx, spiny, spinz, identity = self.transformed_XYZI
        else:
            spinx, spiny, spinz, identity = precomputed_XYZI_transform

        # 3. Map the individual spin S systems in the register to qubits.
        # Go through the register of spin systems and construct the embedded qubit
        # operator `embed(X^nx * Y^ny * Z^nz)`. Add this embedded operator to `operatorlist`.
        for nx, ny, nz in zip(self.Sx, self.Sy, self.Sz):
            operator_on_spin_i = []
            if nx > 0:
                # construct the qubit operator embed(X^nx)
                operator_on_spin_i.append(operator_product([spinx for i in range(nx)]))
            if ny > 0:
                # construct the qubit operator embed(Y^ny)
                operator_on_spin_i.append(operator_product([spiny for i in range(ny)]))
            if nz > 0:
                # construct the qubit operator embed(Z^nz)
                operator_on_spin_i.append(operator_product([spinz for i in range(nz)]))
            if np.any([nx, ny, nz]) > 0:
                # multiply X^nx * Y^ny * Z^nz
                operator_on_spin_i = operator_product(operator_on_spin_i)
                # get rid of vanishing paulistrings
                operator_on_spin_i.chop()
                # Add the embedded qubit operator acting on spin system `i` to the operatorlist,
                # to then form a tensorproduct in the next step.
                operatorlist.append(operator_on_spin_i)
            else:
                # If nx=ny=nz=0, simply add the embedded Identity operator.
                operatorlist.append(identity)

        # `operatorlist` is now a list of (sums of pauli strings) which still need to be tensored together
        # to get the final operator

        # 4. Tensor the individual embedded operators together:
        # Extract the pauli strings from the individual site (individual spin S system) aqua operators

        paulis_per_site = [site_operator.paulis for site_operator in operatorlist]

        # Generate all possible combinations of tensor products of pauli strings which make up
        # the embedded operator. (I.e. the summand pauli strings of the final operator)
        combos = list(itertools.product(*paulis_per_site))
        # Tensor the paulis together
        tensored_paulis = [tensor_paulis(combo, multiply_coeff=self.coeff) for combo in combos]

        # Construct the final operator as a sum of these pauli strings
        final_operator = WeightedPauliOperator(paulis=tensored_paulis)

        return final_operator

    def to_qubit_operator(self, output='qiskit', **kwargs):
        """
        Wrapper function for the conversion of a `SpinSOperator` to an operator that acts on qubits.
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
