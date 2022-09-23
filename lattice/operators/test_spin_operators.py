import numpy.random as rand
import time
from operators.spin_operators import *


def random_spin_operator(S, register_length):
    # Generate random strings of XYZ operators
    Sx, Sy, Sz = [rand.poisson(0.5, size=register_length) for i in range(3)]
    # Generate random coefficient
    coeff = round(rand.normal(), 3) + 1j * round(rand.normal(), 3)

    return BaseSpinOperator(S, Sx=Sx, Sy=Sy, Sz=Sz, coeff=coeff)


# Check qutip matrix versus qiskit matrix for exactly representable systems
print('Test 0: Check qutip matrix versus qiskit matrix for exactly representable systems:')

flag0 = True
for S in [1/2, 3/2]:
    print('------ Checking spin S=%.1f -------' % round(S, 1))
    op = random_spin_operator(S, 1)
    transform_xyz = op._logarithmic_encoding()
    for length in range(1, 4):
        for i in range(11):
            op = random_spin_operator(S, length)

            #t0 = time.process_time()
            qutip_matrix = op.to_matrix()
            #t1 = time.process_time()
            #if t1 - t0 > 1e-1:
            #    print('qutip needed', round(t1 - t0, 2))

            op.transformed_XYZI = transform_xyz
            #t1 = time.process_time()
            qiskit_matrix = op.to_qiskit()
            #t2 = time.process_time()
            #if t2 - t1 > 1e-1:
            #    print('qiskit needed', round(t2 - t1, 2))

            qiskit_matrix.to_matrix()
            qiskit_matrix = qiskit_matrix.matrix
            if len(qiskit_matrix.shape) == 1:
                qiskit_matrix = np.diag(qiskit_matrix)
            else:
                qiskit_matrix = qiskit_matrix.todense()
            # print(qiskit_matrix)

            if not np.allclose(qutip_matrix, qiskit_matrix):
                print('Coeff, Label: %.4f' % (op.coeff.real), op.label, ' \t Test failed.')
                flag0 = False
                break
            #t3 = time.process_time()
            #if t3 - t2 > 1:
            #    print('comparison needed', round(t3 - t2, 1))
        print('\t Test passed for %d random operators of spin S=%.1f and length %d' % (i, round(S, 1), length))


# Check qutip matrix versus qiskit matrix for exactly representable systems
print('Test 1: Check qutip matrix versus qiskit matrix for sums of operators:')
flag1 = True
for S in [1/2, 3/2]:
    print('------ Checking spin S=%.1f -------' % round(S, 2))
    for length in range(1, 4):
        for i in range(11):
            op1 = random_spin_operator(S, length)
            op2 = random_spin_operator(S, length)
            op = op1 + op2

            #t0 = time.process_time()
            qutip_matrix = op.to_matrix()
            #t1 = time.process_time()
            #if t1 - t0 > 1e-1:
            #    print('qutip needed', round(t1 - t0, 2))

            #t1 = time.process_time()
            qiskit_matrix = op.to_qiskit()
            #t2 = time.process_time()
            #if t2 - t1 > 1e-1:
            #    print('qiskit needed', round(t2 - t1, 2))

            qiskit_matrix.to_matrix()
            qiskit_matrix = qiskit_matrix.matrix
            if len(qiskit_matrix.shape) == 1:
                qiskit_matrix = np.diag(qiskit_matrix)
            else:
                qiskit_matrix = qiskit_matrix.todense()
            # print(qiskit_matrix)

            if not np.allclose(qutip_matrix, qiskit_matrix):
                print('Coeff, Label: %.4f' % (op.coeff.real), op.label, ' \t Test failed.')
                flag0 = False
                break
            #t3 = time.process_time()
            #if t3 - t2 > 1:
            #    print('comparison needed', round(t3 - t2, 1))
        print('\t Test passed for %d random operators of spin S=%.1f and length %d' % (i, round(S, 1), length))