import time
from qiskit import Aer, execute, QuantumRegister, QuantumCircuit
from qiskit.aqua.operators import evolution_instruction
from qiskit.aqua.components.initial_states import Zero, Custom
from ..operators.qiskit_aqua_operator_utils import operator_sum
import numpy as np
import matplotlib.pyplot as plt
from qiskit.aqua.operators.legacy.op_converter import to_matrix_operator
from .states import *

################################################################################################
# 1. Utils for constructing trotter circuits
################################################################################################


def construct_trotter_step(operator, t_stepsize, name=None, use_basis_gates=False, num_time_slices=1 ):
    """
    Wrapper to create a single Trotter step from a given hermitean operator in paulis representation.

    Args:
        operator (qiskit.aqua.operators.WeightedPauliOperator): The qiskit.aqua operator to be trotterized
                                                                in paulis representation.
        t_stepsize (float): A float giving the parameter in the exponential in the trotterization.

    Returns:
        qiskit.circuit.Instruction
    """
    # Set up the circuit for one trotter step
    #di una withged pauli matrix toglie tutti i contributi che sono minori del valore che gli si mette tra parentesi
    #qunado non c'é l'argomento non so cosa voglia dir, magari toglie quelli piccolissimi 
    #toglie i contrubuti immaginari uguali a 0 tipo 0j
    operator.chop()
    trotter_step = evolution_instruction(operator.paulis, t_stepsize, num_time_slices=num_time_slices, use_basis_gates=use_basis_gates)

    if name is None:
        trotter_step.name = 'trotter_step({})'.format(t_stepsize)
    else:
        trotter_step.name = name

    return trotter_step


def trotter_circuit(Hamiltonian, init_circuit, t_stepsize=0.1, nsteps=3):
    """
    Wrapper to create a Trotter QuantumCircuit object from a Hamiltonian and initial state
    in qiskit.

    Args:
        Hamiltonian (qiskit.aqua.operators.WeightedPauliOperator): A WeightedPauliOperator representing the Hamiltonian
            to be trotterized.
        init_circuit (qiskit.QuantumCircuit): The initial state which should be time-evolved
            with the Trotter circuit.
        t_stepsize (float): The stepsize of one single trotter step
        nsteps (int): The number of trotter steps in the circuit.

    Returns:
        qiskit.QuantumCircuit
    """
    # Set up the registers and the circuit:
    qregs = init_circuit.qregs
    evolution_circuit = QuantumCircuit(*qregs)

    all_qubits = operator_sum([[q for q in qreg] for qreg in evolution_circuit.qregs])

    # Construct the initialization gate
    init = init_circuit.to_instruction()
    init.name = 'init'
    evolution_circuit.append(init, qargs=all_qubits)

    # Construct the circuit for one trotter step
    trotter_step = construct_trotter_step(operator=Hamiltonian, t_stepsize=t_stepsize, use_basis_gates=False)

    # Build up the full evolution circuit

    for step in range(nsteps):
        evolution_circuit.append(trotter_step, qargs=all_qubits)

    return evolution_circuit


#nota ceh il trotter completo va cosi:
#praticamente non usa trotter construct 
# fa un trotter step, salva lo stato in una lista, poi prende l'utimo stato evoluto e fa un altro step, e cosi via
#salva tutto nella lista per il plot finale 
def qiskit_trotter_simulate(Hamilton, init_state, t_stepsize=0.1, nsteps=10, qiskit_trotter_matrix=None):
    """
    Simulating the qiskit trotter evolution by first getting the full unitary circuit matrix and then
    taking powers of it.
    """
    print('1. Constructing trotter step')
    if qiskit_trotter_matrix is None:
        # 1. build circuit for one trotter step
        trotter_step = construct_trotter_step(Hamilton, t_stepsize)

        print('2. Extracting trotter matrix')
        # 2. extract the matrix for one trotter step from unitary simulation
        circ = Zero(Hamilton.num_qubits).construct_circuit(mode='circuit')
        circ.append(trotter_step, qargs=circ.qubits)

        unitary_backend_cpp = Aer.get_backend('unitary_simulator')
        trotter_result = execute(circ, unitary_backend_cpp).result()
        #praticamente é un modo intelligente, per ottenere la matrice dell'hamiltoniano 
        #esponenziato e moltiplicato con un certo time step, é per avere la matrice unitaria derivante dal circuit!!
        qiskit_trotter_matrix = trotter_result.get_unitary()

    print('3. Running time evolution')
    # 3. run the time evolution
    #nota il flatten prende [[1],[2]] e fa [1,2]
    if isinstance(init_state, DiracState):
        init_vec = init_state.construct_circuit(mode='vector').flatten()
    if isinstance(init_state, Custom):
        init_vec = init_state.construct_circuit(mode='vector').flatten()    
    if isinstance(init_state, np.ndarray):
        init_vec = init_state
    # set up an array to save the intermediate states in memory
    #fa ancora flatten ma penso non ci sarebbe bisogno 
    #evolved_state sara un array con dentro tanti vettori 
    #[[v1],[v2],...] ognuno che rappresenta lo stato alla fine 
    #dello step !! 
    evolved_states = [init_vec.flatten()]

    # perform the trotter evolution and save the interediate states
    #é il vero tempo del PC
    t0 = time.time()
    for step in range(1, nsteps + 1):
        ti0 = time.time()

        # set up the circuit for one additional trotter step starting from the current state
        current_state = evolved_states[-1]

        # generate next state
        #attenzione potevo fare tutto anche con state vector,
        #l'unica cosa é che facendo con state vector, avrei dovuto 
        #ricreare ad ogni step un nuovo circuito con l'init vector
        #che fosse uguale allo state vecotr del passaggio precedente
        next_state = qiskit_trotter_matrix @ current_state
        # next_state = np.dot(qiskit_trotter_matrix, current_state)

        # append it to state list
        evolved_states.append(next_state)

        # print status output
        ti1 = time.time()
        delta_time = ti1 - ti0
        if step == 1: print('forcasted total time: {:.1f}s'.format(nsteps * delta_time))
        print('completed', step, 'of', nsteps,
              'in time {:.3f}s   -   {:.1f}s left'.format(delta_time, (nsteps - step) * delta_time), end='\r')

    print('\ntotal time elapsed: {:.3}s'.format(ti1 - t0))
    tfin = time.time()
    print('TOTAL TIME: {:.3f}s'.format(tfin - t0))

    times = np.arange(0, t_stepsize * nsteps + t_stepsize, t_stepsize)

    return times, evolved_states, qiskit_trotter_matrix


################################################################################################
# 2. Utils for measuring expecation values in simulation
################################################################################################

def expectation_val(matrix, vector):
    """Calculates the expectation value, i.e. vector.H @ matrix @ vector.
    Note: matrix must be a non-sparse numpy array."""
    if len(matrix.shape) == 1:
        # Treat the case of a diagonal matrix
        assert matrix.shape == vector.shape
        return np.inner(vector.conj(), matrix * vector)

    return np.inner(vector.conj(), np.dot(matrix, vector))


def qiskit_overlap(state1, state2):
    """Calculates the overlap of two state vectors`state1` and `state2`"""
    return np.inner(state1.conj(), state2)

### attenzione modifico prende observable come weighted pauli 
def qiskit_expectation(observable, Diracstate):
    """Calculates the expectation value of `obeservable` in `state`."""
    observable=to_matrix_operator(observable)
    observable=observable.dense_matrix
    state=np.squeeze(Diracstate.construct_circuit("vector"))
    return expectation_val(observable, state)

################################################################################################
# 3. Utils for plotting the evolution of states
################################################################################################


def plot_qiskit_evolution(times, value, **kwargs):
    """Plots the values vs. times and fills below"""
    plt.plot(times, value, **kwargs)
    plt.fill_between(times, value, alpha=0.2)
    plt.xlabel('Time')
