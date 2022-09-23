import copy
import numbers
import itertools
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator, TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.legacy.op_converter import *
import numpy as np
import matplotlib.pyplot as plt
import warnings


def operator_sum(op_list):
    """Calculates the sum of all elements of a non-empty list

    Args:
        op_list (list):
            The list of objects to sum, i.e. [obj1, obj2, ..., objN]

    Returns:
        obj1 + obj2 + ... + objN
    """
    assert len(op_list) > 0, 'Operator list must be non-empty'

    if len(op_list) == 1:
        return copy.deepcopy(op_list[0])
    else:
        op_sum = copy.deepcopy(op_list[0])
        for elem in op_list[1:]:
            op_sum += elem
    return op_sum


def operator_product(op_list):
    """
    Calculates the product of all elements in a non-empty list.

    Args:
        op_list (list):
            The list of objects to sum, i.e. [obj1, obj2, ..., objN]

    Returns:
        obj1 * obj2 * ... * objN

    """
    assert len(op_list) > 0, 'Operator list must be non-empty'

    if len(op_list) == 1:
        return op_list[0]
    else:
        op_prod = op_list[0]
        for elem in op_list[1:]:
            op_prod *= elem
    return op_prod


def tensor_paulis(paulilist, multiply_coeff = 1.):
    """
    Tensors together a list of Pauli string operators with coefficients.

    Args:
        paulilist (list/np.ndarray):
            A list of the form [(coeff1, paulis1), (coeff2, paulis2), ...], where the coeff's are complex
            numbers and the paulis are qiskit.quantum_info.Pauli objects.
        multiply_coeff (complex):
            An overall coefficient with which to multiply the resulting paulistring

    Returns:
        total_coeff (complex):
            The product of all coefficients, i.e. (mulitply_coeff * coeff1 * coeff2 * ... )
        total_paulis (qiskit.quantum_info.Pauli):
            The tensored paulistring corresponding to (paulis1 \otimes paulis2 \otimes ... )

    """

    total_coeff  = copy.deepcopy(paulilist[0][0]) * multiply_coeff  # multiply an overall coefficient
    total_paulis = copy.deepcopy(paulilist[0][1])                   # extract first pauli to build total paulis

    for coeff, paulis in paulilist[1:]:
        # Parse coeff, paulis
        if not isinstance(coeff, numbers.Number):
            raise TypeError("`coeff` must be a number type object, not '{}'").format(type(coeff).__name__)
        if not isinstance(paulis, Pauli):
            raise TypeError("`paulilist` must be a list of tuples (coeff, pauli) where pauli is"
                            " a qiskit.quantum_info.Pauli object, not '{}").format(type(paulis).__name__)

        # Multiply coefficient and append paulistring to the total paulistring
        total_coeff *= coeff
        total_paulis.append_paulis(paulis=paulis)

    return [total_coeff, total_paulis]


def tensor_aqua_operators(aqua_operatorlist):
    """
    Computes the tensor product of a list of qiskit.aqua.operators WeightedPauliOperator objects
    in `pauli` representation.
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


def to_qiskit_matrix(operator):
    """
    Transforms a qiskit.aqua.operators WeightedPauliOperator, MatrixOperator or TPBGroupedWeightedPauliOperator
    into a dense numpy matrix.

    Args:
        operator (qiskit.aqua.operators):
            The qiskit.aqua.operators Operator to transform.

    Returns:
        np.ndarray
    """
    assert isinstance(operator, (WeightedPauliOperator, MatrixOperator, TPBGroupedWeightedPauliOperator)), \
        '`operator` must be an object of type `WeightedPauliOperator`, `MatrixOperator` or ' \
        '`TPBGroupedWeightedPauliOperator`'

    operator = to_matrix_operator(operator)

    matrix = operator.matrix
    if len(matrix.shape) == 1:
        matrix = np.diag(matrix)
    else:
        matrix = matrix.todense()
    return matrix


def check_qiskit_vs_qutip(operator):
    """Checks whether the matrix constructed from qiskit and that from qutip agree."""
    qiskit_matrix = to_qiskit_matrix(operator.to_qubit_operator())
    qutip_matrix = operator.to_qubit_operator(output='matrix')
    return np.allclose(qiskit_matrix, qutip_matrix)


def oplot(op, part='real'):
    """Plots an operator (MixedOperator, FermionicOperator, SpinSOperator, BaseFermionOperator, BaseSpinOperator)"""
    operator = op.to_qubit_operator(output='qutip')

    if part == 'real':
        if np.any(np.iscomplex(operator)):
            warnings.warn('Operator has nonzero imaginary part')
        plt.matshow(operator.data.todense().real)

    elif part == 'imag':
        plt.matshow(operator.data.todense().imag)
        if np.any(np.isreal(operator)):
            warnings.warn('Operator has nonzero real part')

    elif part == 'both':
        plt.matshow(operator.data.todense().real)
        plt.colorbar();
        plt.matshow(operator.data.todense().imag)

    plt.colorbar();
