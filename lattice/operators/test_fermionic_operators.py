import random
from operators.fermionic_operators import *


def random_fermionic_label(k, weights=None):
    """
    Generate a random fermionic label

    Args:
        k: int, length of the label
        weights:

    Returns:

    """
    return ''.join(random.choices(['I', 'N', 'E', '+', '-'], weights, k=k))


print('Test 0: Check square of operators containing + or - is 0:')
flag0 = True
for i in range(1,5):
    for j in range(51):
        label = random_fermionic_label(i, weights = [0,0,0,1,1])
        coeff = random.gauss(0, 2)
        op = BaseFermionOperator(label, coeff)

        # Test1: Check that direct JW transform or normal ordering and then JW transform agree.
        op2 = op * op

        if op2.coeff != 0:
            print('Coeff, Label: %.4f' % (coeff), label, ' \t Test failed.')
            flag0 = False
            break
    print('\t Test passed for length %d for %d random operators' % (i, j))


print('Test 1: Check JW Transform vs. Normal Ordering and then JW Transform')
flag1 = True
for i in range(5):
    for j in range(51):
        label = random_fermionic_label(i)
        coeff = random.gauss(0, 2)
        op = BaseFermionOperator(label, coeff)

        # Test1: Check that direct JW transform or normal ordering and then JW transform agree.
        jw_transform = op.jordan_wigner_transform()
        normal_jw_transform = operator_sum([elem.jordan_wigner_transform() for elem in op.normal_order()])
        difference = jw_transform - normal_jw_transform
        difference.chop()

        if difference.paulis != []:
            print('Coeff, Label: %.4f' % (coeff), label, ' \t Test failed.')
            flag1 = False
            break
    print('\t Test passed for length %d for %d random operators' % (i, j))
if flag1:
    print('Test passed')
else:
    print('Test failed')

print('Test 2: Check that the product commutes with the JW transforms')
flag2 = True
for i in range(5):
    for j in range(51):
        label1 = random_fermionic_label(i)
        coeff1 = random.gauss(0, 2)

        label2 = random_fermionic_label(i)
        coeff2 = random.gauss(0, 2)

        op1 = BaseFermionOperator(label1, coeff1)
        op2 = BaseFermionOperator(label2, coeff2)

        # Test2: Check that the JW transform commutes with the product operation
        product_first = (op1 * op2).jordan_wigner_transform()
        jw_first = op1.jordan_wigner_transform() * op2.jordan_wigner_transform()

        difference = product_first - jw_first
        difference.chop()

        if difference.paulis != []:
            print('Labels: {} and {}'.format(label1, label2), ' \t Test failed.')
            flag2 = False
            break
    print('\t Test passed for length %d for %d random operators' % (i, j))
if flag2:
    print('Test passed')
else:
    print('Test failed')

print('Test 3: Check that the dagger operation commutes with the JW transforms')
flag3 = True
for i in range(1,6):
    for j in range(101):
        label = random_fermionic_label(i)
        coeff = random.gauss(0, 2)

        op = BaseFermionOperator(label, coeff)

        # Test2: Check that the JW transform commutes with the product operation
        dag_first = (op.dag()).jordan_wigner_transform()
        jw_first = op.jordan_wigner_transform()

        jw_first._paulis_to_matrix()
        jw_first._matrix = jw_first._matrix.H
        jw_first._matrix_to_paulis()

        difference = dag_first - jw_first
        difference.chop()

        if difference.paulis != []:
            print('Labels {}'.format(label), ' \t Test failed.')
            flag3 = False
            break
    print('\t Test passed for length %d for %d random operators' % (i, j))
if flag2:
    print('Test passed')
else:
    print('Test failed')